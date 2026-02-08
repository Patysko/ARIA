"""
ARIA WebUI — HTTP server with real-time dashboard.

ALL CLI commands work in chat via /slash syntax.

Chain of Thought architecture:
  User message -> Analyze -> Recall memory -> Run skills -> Plan -> Final answer
  Each step is a separate LLM call with structured output format.
"""

import json
import re
import time
import threading
import queue
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.memory import CompressedMemory
from core.skills_manager import SkillsManager
from core.computer import ComputerTools
from core.reflection import ReflectionThread
from core.llm import LLMClient
from core.prompts import (
    get_lang, SYSTEM_PROMPT_TEMPLATE,
    COT_SYSTEM, CHAT_SYSTEM, COT_ANALYZE, COT_PLAN, COT_INTERPRET, COT_FINAL,
    COT_FALLBACK_ANALYZE, COT_FALLBACK_PLAN,
    CMD_DESC, UI,
)


class AriaWebAgent:
    """Headless agent for WebUI — no CLI, just API."""

    def __init__(self, config: Config):
        self.config = config
        self.memory = CompressedMemory(config)
        self.skills = SkillsManager(config)
        self.computer = ComputerTools()
        self.reflection = ReflectionThread(config, self.memory, self.skills)
        self.llm = LLMClient(config.LLM_CONFIG)
        self.reflection_llm = LLMClient(config.REFLECTION_LLM_CONFIG)
        self.llm_connected = False
        self.interaction_count = 0
        self.start_time = time.time()

        self._sse_queues: list[queue.Queue] = []
        self._sse_lock = threading.Lock()
        self._agent_lock = threading.Lock()
        self.chat_history: list[dict] = []

        # Pending messages from Thread 2 to user
        self._pending_messages: list[dict] = []
        self._pending_lock = threading.Lock()

    def initialize(self):
        status = self.llm.check_connection()
        self.llm_connected = status.get("connected", False)
        if self.llm_connected:
            self.reflection.on_thought = self._broadcast_thought
            self.reflection.on_user_message = self._queue_user_message
            self.reflection.start_background(
                llm_client=self.reflection_llm, interval=30
            )
        return status

    def _broadcast_thought(self, thought: str):
        event = {
            "type": "thought", "data": thought,
            "cycle": self.reflection._cycle_count,
            "timestamp": time.time(),
        }
        with self._sse_lock:
            dead = []
            for q in self._sse_queues:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._sse_queues.remove(q)

    def _queue_user_message(self, message: str):
        """Thread 2 can send proactive messages to the user."""
        event = {
            "type": "proactive_message",
            "data": message,
            "timestamp": time.time(),
        }
        with self._pending_lock:
            self._pending_messages.append(event)
        # Also broadcast via SSE so the UI gets it immediately
        with self._sse_lock:
            for q in self._sse_queues:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def get_pending_messages(self) -> list[dict]:
        with self._pending_lock:
            msgs = list(self._pending_messages)
            self._pending_messages.clear()
        return msgs

    def subscribe_sse(self) -> queue.Queue:
        q = queue.Queue(maxsize=100)
        with self._sse_lock:
            self._sse_queues.append(q)
        return q

    def unsubscribe_sse(self, q: queue.Queue):
        with self._sse_lock:
            if q in self._sse_queues:
                self._sse_queues.remove(q)

    # =========================================
    #  COMMAND ROUTER
    # =========================================

    def handle_message(self, message: str) -> dict:
        with self._agent_lock:
            msg = message.strip()
            if msg.startswith("/"):
                return self._route_command(msg)
            else:
                return self._chat(msg)

    def _route_command(self, cmd: str) -> dict:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        routes = {
            "/help":         lambda: self._cmd_help(),
            "/status":       lambda: self._cmd_status(),
            "/memory":       lambda: self._cmd_memory(),
            "/recall":       lambda: self._cmd_recall(arg),
            "/skills":       lambda: self._cmd_skills(),
            "/skill":        lambda: self._cmd_skill_detail(arg),
            "/run":          lambda: self._cmd_run(arg),
            "/create-skill": lambda: self._cmd_create_skill(arg),
            "/thread2":      lambda: self._cmd_thread2(arg),
            "/reflect":      lambda: self._cmd_reflect(),
            "/exec":         lambda: self._cmd_exec(arg),
            "/python":       lambda: self._cmd_python(arg),
            "/ls":           lambda: self._cmd_ls(arg),
            "/read":         lambda: self._cmd_read(arg),
            "/write":        lambda: self._cmd_write(arg),
            "/sysinfo":      lambda: self._cmd_sysinfo(),
            "/compress":     lambda: self._cmd_compress(),
            "/selfmodel":    lambda: self._cmd_selfmodel(),
            "/models":       lambda: self._cmd_models(),
            "/model":        lambda: self._cmd_model(arg),
            "/ollama":       lambda: self._cmd_ollama(),
        }

        handler = routes.get(command)
        if handler:
            result = handler()
            # Commands go to chat history for display but NOT to agent memory
            self.chat_history.append({"role": "user", "content": cmd, "ts": time.time()})
            self.chat_history.append({"role": "assistant", "content": result["content"],
                                      "is_command": True, "ts": time.time()})
            return result
        else:
            return {"type": "error", "content": UI[get_lang()]["unknown_cmd"].format(cmd=command)}

    # ---- Command implementations ----

    def _cmd_help(self) -> dict:
        lang = get_lang()
        cmds = CMD_DESC.get(lang, CMD_DESC["en"])
        lines = [UI[lang]["help_title"]]
        for cmd, desc in cmds.items():
            lines.append(f"`{cmd}` -- {desc}")
        return {"type": "command", "content": "\n".join(lines)}

    def _cmd_status(self) -> dict:
        s = self.get_status()
        sm = s["self_model"]
        conn = "OK" if s["llm_connected"] else "OFFLINE"
        t2 = "aktywny" if s["thread2_running"] else "stop"
        return {"type": "command", "content": (
            f"**ARIA v{s['version']}**\n"
            f"LLM: `{s['llm_model']}` {conn} @ `{s['llm_url']}`\n"
            f"Thread 2: {t2} | cykli: {s['thread2_cycles']}\n"
            f"Pamiec: ST={s['memory']['short_term_count']} LT={s['memory']['long_term_count']} EP={s['memory']['episodic_count']}\n"
            f"Kompresje: {s['memory']['compression_count']} | Umiejetnosci: {s['skills_count']}\n"
            f"Interakcje: {s['interactions']} | Uptime: {s['uptime']}s"
        )}

    def _cmd_memory(self) -> dict:
        d = self.get_memory_dump()
        lines = [f"**Pamiec** ST:{len(d['short_term'])} LT:{len(d['long_term'])}\n"]
        lines.append("**Krotkoterminowa:**")
        for e in d["short_term"][-10:]:
            imp = "*" * round(e["importance"] * 5)
            lines.append(f"  `[{e['category']}]` {imp} {e['content'][:80]}")
        if d["long_term"]:
            lines.append("\n**Skompresowana:**")
            for e in d["long_term"][-5:]:
                lines.append(f"  `[{e['category']}]` ({e['source_count']}src) {e['summary'][:80]}")
        return {"type": "command", "content": "\n".join(lines)}

    def _cmd_recall(self, query) -> dict:
        if not query:
            return {"type": "error", "content": "Uzycie: `/recall <zapytanie>`"}
        results = self.memory.recall(query)
        lines = [f"**Wyniki: \"{query}\"**\n"]
        for label, key in [("Krotkoterminowa", "short_term"), ("Dlugoterminowa", "long_term")]:
            if results[key]:
                lines.append(f"**{label}:**")
                for score, entry in results[key]:
                    c = entry.get("content", entry.get("summary", ""))
                    lines.append(f"  [{score:.1f}] {c[:100]}")
        if not results["short_term"] and not results["long_term"]:
            lines.append("Brak wynikow.")
        return {"type": "command", "content": "\n".join(lines)}

    def _cmd_skills(self) -> dict:
        skills = self.skills.list_all()
        if not skills:
            return {"type": "command", "content": "Brak umiejetnosci. Thread 2 stworzy je automatycznie."}
        lines = [f"**Umiejetnosci ({len(skills)})**\n"]
        for s in skills:
            scripts = ", ".join(s.get("scripts", [])) or "brak"
            lines.append(f"* **{s['name']}** [{s['use_count']}x] -- {s['description'][:60]}\n  Skrypty: `{scripts}`")
        return {"type": "command", "content": "\n".join(lines)}

    def _cmd_skill_detail(self, name) -> dict:
        if not name:
            return {"type": "error", "content": "Uzycie: `/skill <nazwa>`"}
        skill = self.skills.get(name)
        if not skill:
            return {"type": "error", "content": f"Nie znaleziono: \"{name}\"\nDostepne: {', '.join(self.skills.list_names())}"}
        scripts = skill.get_scripts()
        code_preview = ""
        for s in scripts:
            if s.suffix == ".py":
                try:
                    code_preview = f"\n```python\n{s.read_text(errors='replace')[:1500]}\n```"
                except Exception:
                    pass
                break
        return {"type": "command", "content": (
            f"**{skill.name}**\n{skill.description}\n\n"
            f"Skrypty: {', '.join(s.name for s in scripts) or 'brak'} | Uzycia: {skill.use_count}\n\n"
            f"**Instrukcje:**\n{skill.instructions[:500]}"
            f"{code_preview}"
        )}

    def _cmd_run(self, arg) -> dict:
        if not arg:
            return {"type": "error", "content": "Uzycie: `/run <skill> [argumenty]`\nDostepne: " + ", ".join(self.skills.list_names())}
        parts = arg.split(maxsplit=1)
        name = parts[0]
        args = parts[1].split() if len(parts) > 1 else []
        skill = self.skills.get(name)
        if not skill:
            return {"type": "error", "content": f"Nie znaleziono: \"{name}\""}
        scripts = [s for s in skill.get_scripts() if s.suffix == ".py"]
        if not scripts:
            return {"type": "error", "content": f"{name} nie ma skryptow Python"}
        result = skill.run_script(scripts[0].name, args=args)
        if result.get("returncode", 1) == 0:
            return {"type": "command", "content": f"**OK {name}/{scripts[0].name}** (exit 0)\n```\n{result.get('stdout','')[:2000]}\n```",
                    "skill_used": name}
        else:
            err = result.get("stderr", result.get("error", ""))
            return {"type": "error", "content": f"**FAIL {name}/{scripts[0].name}** (exit {result.get('returncode','?')})\n```\n{err[:1000]}\n```"}

    def _cmd_create_skill(self, arg) -> dict:
        if not arg:
            return {"type": "error", "content": (
                "Uzycie: `/create-skill {json}`\n"
                'Przyklad: /create-skill {"name": "hello-world", "description": "Test", '
                '"instructions": "Prints hello", "script_code": "print(\'Hello!\')"}'
            )}
        try:
            data = json.loads(arg)
            name = data.get("name", "")
            desc = data.get("description", "")
            if not name or not desc:
                return {"type": "error", "content": "JSON musi zawierac `name` i `description`"}
            scripts = {}
            if data.get("script_code"):
                scripts[data.get("script_name", "main.py")] = data["script_code"]
            skill = self.skills.create_skill(name, desc, data.get("instructions", desc), scripts=scripts)
            return {"type": "command", "content": f"OK Stworzono umiejetnosc: **{skill.name}**\nOpis: {desc}"}
        except json.JSONDecodeError as e:
            return {"type": "error", "content": f"Niepoprawny JSON: {e}"}

    def _cmd_thread2(self, arg) -> dict:
        n = int(arg) if arg and arg.isdigit() else 5
        refs = self.reflection.get_last_reflections(n)
        running = self.reflection.is_running
        lines = [
            f"**Thread 2** {'aktywny' if running else 'stop'} | "
            f"Cykli: {self.reflection._cycle_count} | Interwal: {self.reflection.interval}s\n"
        ]
        if refs:
            for ref in reversed(refs):
                phase = ref.get("phase", "?")
                cycle = ref.get("cycle", "?")
                ts = time.strftime("%H:%M:%S", time.localtime(ref["timestamp"]))
                lines.append(f"**Cykl {cycle}** | `{phase}` | {ts}")
                for t in ref.get("thoughts", [])[:4]:
                    lines.append(f"  {t[:150]}")
                lines.append("")
        else:
            lines.append("Brak refleksji -- poczekaj chwile.")
        return {"type": "command", "content": "\n".join(lines)}

    def _cmd_reflect(self) -> dict:
        thoughts = self.reflection.reflect(trigger="manual")
        lines = ["**Refleksja (wymuszona)**\n"]
        for t in thoughts:
            lines.append(f"  {t[:200]}")
        return {"type": "command", "content": "\n".join(lines)}

    def _cmd_exec(self, cmd) -> dict:
        if not cmd:
            return {"type": "error", "content": "Uzycie: `/exec <komenda>`"}
        r = self.computer.execute(cmd)
        output = ""
        if r.get("stdout"): output += r["stdout"][:3000]
        if r.get("stderr"): output += "\n**STDERR:**\n" + r["stderr"][:1000]
        if r.get("error"): output += "\n**ERROR:** " + r["error"]
        code = r.get("returncode", "?")
        return {"type": "command", "content": f"**`{cmd}`** (exit {code})\n```\n{output.strip()}\n```"}

    def _cmd_python(self, code) -> dict:
        if not code:
            return {"type": "error", "content": "Uzycie: `/python <kod>`"}
        r = self.computer.run_python(code)
        output = r.get("stdout", "") or r.get("error", "") or r.get("stderr", "")
        return {"type": "command", "content": f"**Python**\n```\n{output[:3000].strip()}\n```"}

    def _cmd_ls(self, path) -> dict:
        r = self.computer.list_dir(path or ".")
        if "error" in r:
            return {"type": "error", "content": r["error"]}
        lines = [f"**{r['path']}**\n"]
        for item in r["items"]:
            icon = "[DIR]" if item["type"] == "dir" else "[FILE]"
            size = f" ({item['size']}B)" if item.get("size") else ""
            lines.append(f"  {icon} {item['name']}{size}")
        return {"type": "command", "content": "\n".join(lines)}

    def _cmd_read(self, path) -> dict:
        if not path:
            return {"type": "error", "content": "Uzycie: `/read <sciezka>`"}
        r = self.computer.read_file(path)
        if "error" in r:
            return {"type": "error", "content": r["error"]}
        return {"type": "command", "content": f"**{r['path']}** ({r['size']}B)\n```\n{r['content'][:4000]}\n```"}

    def _cmd_write(self, arg) -> dict:
        if not arg:
            return {"type": "error", "content": "Uzycie: `/write <sciezka> <tresc>`"}
        parts = arg.split(maxsplit=1)
        if len(parts) < 2:
            return {"type": "error", "content": "Uzycie: `/write <sciezka> <tresc>`"}
        path, content = parts
        r = self.computer.write_file(path, content)
        if r.get("success"):
            return {"type": "command", "content": f"OK Zapisano {r['size']}B do `{r['path']}`"}
        return {"type": "error", "content": r.get("error", "Blad zapisu")}

    def _cmd_sysinfo(self) -> dict:
        info = self.computer.system_info()
        lines = ["**System**\n"]
        for k, v in info.items():
            lines.append(f"  `{k}`: {v}")
        return {"type": "command", "content": "\n".join(lines)}

    def _cmd_compress(self) -> dict:
        r = self.memory.compress()
        return {"type": "command", "content": f"Skompresowano {r['compressed']} -> {r['blocks_created']} blokow"}

    def _cmd_selfmodel(self) -> dict:
        m = self.reflection.self_model
        lines = ["**Model Wlasny**\n"]
        for k, v in m.items():
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v[:5])
            lines.append(f"  `{k}`: {v}")
        return {"type": "command", "content": "\n".join(lines)}

    def _cmd_models(self) -> dict:
        models = self.llm.list_models()
        if not models:
            return {"type": "error", "content": "Nie mozna pobrac listy modeli. Ollama dostepna?"}
        current = self.llm.model
        lines = [f"**Modele Ollama ({len(models)})**\n"]
        for m in models:
            marker = " <-- aktywny" if m == current else ""
            lines.append(f"  * `{m}`{marker}")
        lines.append(f"\nZmien model: `/model <nazwa>`")
        return {"type": "command", "content": "\n".join(lines)}

    def _cmd_model(self, name) -> dict:
        """Switch model. Auto-pulls if not available locally."""
        if not name:
            return {"type": "error", "content": f"Uzycie: `/model <nazwa>`\nAktywny: `{self.llm.model}`\nDostepne: `/models`"}

        old = self.llm.model
        lines = []

        # Check if model is available, pull if not
        if not self.llm.is_model_available(name):
            lines.append(f"Model `{name}` nie znaleziony lokalnie. Pobieram...")
            try:
                for progress in self.llm.pull_model(name):
                    lines.append(progress.strip())
            except Exception as e:
                lines.append(f"Blad pobierania: {e}")
                return {"type": "error", "content": "\n".join(lines)}

        self.llm.model = name
        self.reflection_llm.model = name
        self.llm.clear_history()

        status = self.llm.check_connection()
        self.llm_connected = status.get("connected", False)

        if not self.reflection.is_running and self.llm_connected:
            self.reflection.on_thought = self._broadcast_thought
            self.reflection.on_user_message = self._queue_user_message
            self.reflection.start_background(llm_client=self.reflection_llm, interval=30)
            lines.append("Thread 2 uruchomiony!")

        try:
            data = json.loads(self.config.CONFIG_FILE.read_text())
            data["llm"]["model"] = name
            data["reflection_llm"]["model"] = name
            self.config.CONFIG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=4))
        except Exception:
            pass

        lines.append(f"OK Model zmieniony: `{old}` -> **`{name}`**\nHistoria czatu wyczyszczona.")
        return {"type": "command", "content": "\n".join(lines)}

    def _cmd_ollama(self) -> dict:
        status = self.llm.check_connection()
        self.llm_connected = status.get("connected", False)
        if status["connected"]:
            return {"type": "command", "content": (
                f"**Ollama** OK\n"
                f"URL: `{status['url']}`\n"
                f"Model: `{status['current_model']}` {'OK' if status['model_available'] else 'NIE ZNALEZIONY'}\n"
                f"Modele: {', '.join(status['models'][:10])}"
            )}
        return {"type": "error", "content": (
            f"**Ollama** OFFLINE - Brak polaczenia z `{status['url']}`\n"
            f"Uruchom: `ollama serve`"
        )}

    # =========================================
    #  CHAT -- Chain of Thought reasoning
    # =========================================

    def _chat(self, message: str) -> dict:
        self.interaction_count += 1

        # Store in memory -- only meaningful user inputs (filtered by memory module)
        cat = self._categorize(message)
        imp = self._assess_importance(message)
        self.memory.add(message, category=cat, importance=imp,
                        metadata={"type": "user_input", "n": self.interaction_count})
        self.chat_history.append({"role": "user", "content": message, "ts": time.time()})

        if not self.llm_connected:
            return self._chat_offline(message)

        self._update_system_prompt()

        # === CHAIN OF THOUGHT ===
        thinking_steps = []
        skill_results = {}
        skills_used = []

        # -- Step 1: Analyze query (LLM call #1) --
        analysis = self._cot_step_analyze(message)
        thinking_steps.append(("Analiza", analysis))

        # -- Step 2: Recall relevant memory (exclude Thread 2 data) --
        memory_ctx = self.memory.get_relevant_context(
            message,
            max_tokens=min(800, self.llm.get_available_context() // 4),
            exclude_sources=["thread2"],
        )
        if memory_ctx and memory_ctx != "(brak powiazanych wspomnien)":
            thinking_steps.append(("Pamiec", memory_ctx))

        # -- Step 3: Find and run relevant skills --
        all_relevant = self.skills.find_relevant(message)
        if all_relevant:
            for skill in all_relevant[:3]:
                scripts = [s for s in skill.get_scripts() if s.suffix == ".py"]
                if scripts:
                    args = self._infer_skill_args(skill, message)
                    result = skill.run_script(scripts[0].name, args=args)
                    if result.get("returncode", 1) == 0 and result.get("stdout"):
                        output = result["stdout"][:1500]
                        skill_results[skill.name] = output
                        skills_used.append(skill.name)
                        thinking_steps.append((
                            f"Skill: {skill.name}",
                            f"args={args}\nWynik: {output[:500]}"
                        ))
                    elif result.get("stderr") or result.get("error"):
                        err = result.get("stderr", result.get("error", ""))[:200]
                        thinking_steps.append((
                            f"Skill ERROR: {skill.name}",
                            f"Blad (args={args}): {err}"
                        ))

        # -- Step 4: Plan response (LLM call #2) --
        plan = self._cot_step_plan(message, analysis, thinking_steps, skill_results)
        thinking_steps.append(("Plan", plan))

        # -- Step 5: Optional additional reasoning --
        needs_more = self._cot_step_needs_more(message, plan, skill_results)
        if needs_more:
            thinking_steps.append(("Dodatkowe", needs_more))

        # -- Final: Generate answer (LLM call #3+) --
        reply = self._cot_step_final_answer(message, thinking_steps, skill_results, memory_ctx)
        reply = self._strip_internal_tags(reply)

        # Build thinking text for UI
        thinking_text = ""
        for label, content in thinking_steps:
            thinking_text += f"**{label}**\n{content}\n\n"

        skill_used = skills_used[0] if skills_used else None

        # Store only brief response summary in memory
        self.memory.add(
            f"[Odpowiedz na: {message[:60]}] {reply[:140]}",
            category="agent_response", importance=0.3,
            metadata={"type": "agent_response", "n": self.interaction_count,
                      "skills_used": skills_used}
        )

        self.chat_history.append({
            "role": "assistant", "content": reply,
            "skill_used": skill_used, "skills_used": skills_used,
            "thinking": thinking_text.strip(),
            "ts": time.time(),
        })

        return {
            "type": "chat", "content": reply,
            "skill_used": skill_used,
            "skills_used": skills_used,
            "thinking": thinking_text.strip(),
        }

    def _chat_offline(self, message: str) -> dict:
        relevant = self.skills.find_relevant(message)
        if relevant:
            skill = relevant[0]
            scripts = [s for s in skill.get_scripts() if s.suffix == ".py"]
            if scripts:
                args = self._infer_skill_args(skill, message)
                result = skill.run_script(scripts[0].name, args=args)
                if result.get("returncode", 1) == 0 and result.get("stdout"):
                    reply = f"[Wynik {skill.name}]\n{result['stdout'][:2000]}"
                    self.chat_history.append({
                        "role": "assistant", "content": reply,
                        "skill_used": skill.name, "ts": time.time(),
                    })
                    return {"type": "chat", "content": reply,
                            "skill_used": skill.name, "thinking": None}
        reply = UI[get_lang()]["offline"]
        self.chat_history.append({"role": "assistant", "content": reply, "ts": time.time()})
        return {"type": "chat", "content": reply, "skill_used": None, "thinking": None}

    # =========================================
    #  CHAIN OF THOUGHT -- Role-specific prompts
    # =========================================

    def _cot_step_analyze(self, message: str) -> str:
        lang = get_lang()
        skills_list = ", ".join(self.skills.list_names()) or "none"
        prompt = COT_ANALYZE[lang].format(skills=skills_list, message=message)
        try:
            self.llm.set_system_prompt(COT_SYSTEM[lang])
            result = self.llm.chat(prompt, include_history=False)
            return self._strip_internal_tags(result)[:600]
        except Exception:
            return COT_FALLBACK_ANALYZE[lang]

    def _cot_step_plan(self, message: str, analysis: str,
                       steps: list, skill_results: dict) -> str:
        lang = get_lang()
        parts = [f"Query: \"{message}\"" if lang == "en" else f"Zapytanie: \"{message}\"",
                 f"Analysis: {analysis[:300]}" if lang == "en" else f"Analiza: {analysis[:300]}"]
        if skill_results:
            for name, output in skill_results.items():
                parts.append(f"Skill {name}: {output[:300]}")
        context = "\n".join(parts)
        prompt = COT_PLAN[lang].format(context=context)
        try:
            self.llm.set_system_prompt(COT_SYSTEM[lang])
            result = self.llm.chat(prompt, include_history=False)
            return self._strip_internal_tags(result)[:600]
        except Exception:
            return COT_FALLBACK_PLAN[lang]

    def _cot_step_needs_more(self, message: str, plan: str,
                             skill_results: dict) -> str:
        lang = get_lang()
        missing_key = "MISSING: nothing" if lang == "en" else "BRAKUJE: nic"
        if missing_key.lower() in plan.lower():
            return ""
        if not skill_results:
            return ""
        results_str = ""
        for name, output in skill_results.items():
            results_str += f"[{name}]: {output[:400]}\n"
        prompt = COT_INTERPRET[lang].format(message=message, results=results_str)
        try:
            self.llm.set_system_prompt(COT_SYSTEM[lang])
            result = self.llm.chat(prompt, include_history=False)
            return self._strip_internal_tags(result)[:400]
        except Exception:
            return ""

    def _cot_step_final_answer(self, message: str, thinking_steps: list,
                                skill_results: dict, memory_ctx: str) -> str:
        lang = get_lang()
        parts = []
        if skill_results:
            lbl = "Executed skill results:" if lang == "en" else "Wyniki uruchomionych umiejetnosci:"
            parts.append(lbl)
            for sname, sout in skill_results.items():
                parts.append(f"[{sname}]:\n{sout[:800]}")

        no_mem = "(no relevant memories)" if lang == "en" else "(brak powiazanych wspomnien)"
        if memory_ctx and memory_ctx != no_mem:
            lbl = "Memory context:" if lang == "en" else "Kontekst z pamieci:"
            parts.append(f"{lbl}\n{memory_ctx[:500]}")

        for label, content in thinking_steps:
            if "Plan" in label:
                lbl = "Response plan:" if lang == "en" else "Plan odpowiedzi:"
                parts.append(f"{lbl} {content[:300]}")
                break

        parts.append(COT_FINAL[lang].format(message=message))
        enriched = "\n".join(parts)

        available = self.llm.get_available_context()
        if self.llm.estimate_tokens(enriched) > available:
            enriched = self._trim_to_budget(enriched, available)

        try:
            self.llm.set_system_prompt(CHAT_SYSTEM[lang])
            return self.llm.chat(enriched)
        except Exception as e:
            return f"[Error: {e}]"

    def _trim_to_budget(self, text: str, max_tokens: int) -> str:
        lines = text.split("\n")
        while self.llm.estimate_tokens("\n".join(lines)) > max_tokens and len(lines) > 5:
            lengths = [(len(l), i) for i, l in enumerate(lines) if 2 < i < len(lines) - 2]
            if lengths:
                lengths.sort(reverse=True)
                lines.pop(lengths[0][1])
            else:
                lines.pop(len(lines) // 2)
        return "\n".join(lines)

    def _infer_skill_args(self, skill, message: str) -> list:
        args = []
        try:
            instructions = skill.instructions or ""
            usage_match = re.findall(
                r'(?:python3?\s+\S+\.py|scripts/\S+\.py)\s+(.+?)(?:\n|$)',
                instructions
            )
            if usage_match:
                example = usage_match[0]
                path_match = re.search(r'[/~]\S+|(?:\w:[\\/]\S+)', message)
                if path_match and '<' in example:
                    args.append(path_match.group())
        except Exception:
            pass
        return args

    def _strip_internal_tags(self, text: str) -> str:
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'<analysis>.*?</analysis>\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'<plan>.*?</plan>\s*', '', text, flags=re.DOTALL)
        return text.strip()

    # =========================================
    #  DATA GETTERS
    # =========================================

    def get_status(self) -> dict:
        stats = self.memory.get_stats()
        model = self.reflection.self_model
        return {
            "version": model.get("version", "1.0.0"),
            "llm_connected": self.llm_connected,
            "llm_model": self.llm.model,
            "llm_url": self.llm.base_url,
            "thread2_running": self.reflection.is_running,
            "thread2_cycles": self.reflection._cycle_count,
            "memory": stats,
            "skills_count": len(self.skills.list_names()),
            "interactions": self.interaction_count,
            "uptime": int(time.time() - self.start_time),
            "self_model": dict(model),
            "language": get_lang(),
        }

    def get_reflections(self, n=20):
        return self.reflection.get_last_reflections(n)

    def get_skills_list(self):
        return self.skills.list_all()

    def get_memory_dump(self):
        st = [{"content": e.content, "category": e.category,
               "importance": e.importance, "ts": e.timestamp}
              for e in self.memory.short_term[-30:]]
        lt = [{"summary": b.summary, "category": b.category,
               "keywords": b.keywords, "source_count": b.source_count}
              for b in self.memory.long_term[-20:]]
        return {"short_term": st, "long_term": lt, "stats": self.memory.get_stats()}

    def _update_system_prompt(self):
        lang = get_lang()
        template = SYSTEM_PROMPT_TEMPLATE.get(lang, SYSTEM_PROMPT_TEMPLATE["en"])
        model = self.reflection.self_model

        available = self.llm.context_length - self.llm.max_tokens
        memory_budget = min(800, available // 4)
        skills_budget = min(500, available // 6)

        skills_section = self.skills.get_skills_prompt_section()
        if self.llm.estimate_tokens(skills_section) > skills_budget:
            skills_section = skills_section[:skills_budget * 4]

        memory_context = self.memory.get_context_summary(max_tokens=memory_budget)

        prompt = template.format(
            version=model.get("version", "1.0.0"),
            personality=model.get("personality", "analityczny"),
            goals=", ".join(model.get("goals", [])[:3]),
            skills_section=skills_section,
            memory_context=memory_context,
        )
        self.llm.set_system_prompt(prompt)

    def _categorize(self, text):
        t = text.lower()
        for cat, kws in {
            "memory": ["pamiec", "zapamietaj", "przypomnij"],
            "skill": ["umiejetnosc", "skill", "naucz"],
            "code": ["kod", "python", "script"],
            "file": ["plik", "file", "zapisz"],
            "system": ["system", "info", "status"],
            "search": ["szukaj", "wyszukaj", "search", "znajdz"],
        }.items():
            if any(kw in t for kw in kws):
                return cat
        return "general"

    def _assess_importance(self, text):
        t = text.lower()
        if any(w in t for w in ["zapamietaj", "wazne", "remember"]):
            return 0.9
        if t in ("test", "hej", "czesc", "hello"):
            return 0.2
        return 0.5

    def shutdown(self):
        self.reflection.stop_background()
        self.memory.save()
        self.config.save_self_model(self.reflection.self_model)


# =========================================
#  HTTP HANDLER
# =========================================

agent: AriaWebAgent = None


class AriaHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors()
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self._serve_static(path)
        elif path == "/api/status":
            self._json_response(agent.get_status())
        elif path == "/api/reflections":
            n = int(params.get("n", [20])[0])
            self._json_response(agent.get_reflections(n))
        elif path == "/api/skills":
            self._json_response(agent.get_skills_list())
        elif path == "/api/memory":
            self._json_response(agent.get_memory_dump())
        elif path == "/api/chat-history":
            self._json_response(agent.chat_history[-50:])
        elif path == "/api/models":
            self._json_response(agent.llm.list_models())
        elif path == "/api/pending":
            self._json_response(agent.get_pending_messages())
        elif path == "/api/events":
            self._handle_sse()
        else:
            self._serve_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length > 0 else {}

        if path in ("/api/message", "/api/chat"):
            msg = body.get("message", "")
            if not msg:
                self._json_response({"error": "No message"}, 400)
                return
            result = agent.handle_message(msg)
            self._json_response(result)
        elif path == "/api/run-skill":
            name = body.get("skill", "")
            result = agent.handle_message(f"/run {name}")
            self._json_response(result)
        elif path == "/api/reflect":
            result = agent.handle_message("/reflect")
            self._json_response(result)
        elif path == "/api/model":
            new_model = body.get("model", "")
            result = agent.handle_message(f"/model {new_model}")
            self._json_response(result)
        else:
            self.send_error(404)

    def _handle_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors()
        self.end_headers()
        q = agent.subscribe_sse()
        try:
            while True:
                try:
                    event = q.get(timeout=15)
                    data = json.dumps(event, ensure_ascii=False)
                    self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            agent.unsubscribe_sse(q)


    def _serve_static(self, filepath: str):
        """Serve static files from web/static/ directory."""
        static_dir = Path(__file__).parent / "static"
        safe = filepath.lstrip("/")
        if not safe:
            safe = "index.html"
        full_path = static_dir / safe

        if not full_path.exists() or not full_path.is_file():
            full_path = static_dir / "index.html"
            if not full_path.exists():
                self.send_error(404)
                return

        ext = full_path.suffix.lower()
        mime_map = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".jsx": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
            ".woff2": "font/woff2",
            ".woff": "font/woff",
        }
        content_type = mime_map.get(ext, "application/octet-stream")
        body = full_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control", "no-cache" if ext == ".html" else "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    global agent
    port = int(os.environ.get("ARIA_PORT", 8080))
    config = Config()
    agent = AriaWebAgent(config)

    print(f"\n  ARIA WebUI starting...")
    status = agent.initialize()
    if status.get("connected"):
        print(f"  Ollama: {agent.llm.model} @ {agent.llm.base_url}")
        print(f"  Thread 2: AKTYWNY (co {agent.reflection.interval}s)")
    else:
        print(f"  Ollama niedostepna - tryb offline")
    print(f"  http://localhost:{port}")
    print(f"  Ctrl+C aby zatrzymac\n")

    server = ThreadedHTTPServer(("0.0.0.0", port), AriaHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Zatrzymuje...")
        agent.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()