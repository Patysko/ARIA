"""
ARIA Agent -- Main CLI orchestrator.
For WebUI, see web/server.py.
All strings via core.prompts for PL/EN support.
"""

import json
import sys
import re
import time
import threading
from pathlib import Path

from core.config import Config
from core.memory import CompressedMemory
from core.skills_manager import SkillsManager
from core.computer import ComputerTools
from core.reflection import ReflectionThread
from core.llm import LLMClient
from core.prompts import get_lang, SYSTEM_PROMPT_TEMPLATE, CMD_DESC, UI


class C:
    PURPLE = "\033[95m"; BLUE = "\033[94m"; CYAN = "\033[96m"
    GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
    BOLD = "\033[1m"; DIM = "\033[2m"; END = "\033[0m"


def colored(text, color):
    return f"{color}{text}{C.END}"


class AriaAgent:
    def __init__(self, config: Config):
        self.config = config
        self.memory = CompressedMemory(config)
        self.skills = SkillsManager(config)
        self.computer = ComputerTools()
        self.reflection = ReflectionThread(config, self.memory, self.skills)
        self.interaction_count = 0
        self.show_reflection = False
        self.llm = LLMClient(config.LLM_CONFIG)
        self.reflection_llm = LLMClient(config.REFLECTION_LLM_CONFIG)
        self.llm_connected = False
        self.stream = config.STREAM_RESPONSES
        self._thought_buffer: list[str] = []
        self._thought_lock = threading.Lock()
        self._thought_count = 0

    @property
    def _ui(self):
        return UI.get(get_lang(), UI["en"])

    @property
    def _cmds(self):
        return CMD_DESC.get(get_lang(), CMD_DESC["en"])

    def start(self, show_reflection: bool = False):
        self.show_reflection = show_reflection
        lang = get_lang()
        print(colored("\n  +===============================================+", C.PURPLE))
        print(colored("  |          ARIA Agent                           |", C.PURPLE))
        print(colored("  |   Autonomous Reflective Intelligence Agent    |", C.PURPLE))
        print(colored("  +===============================================+\n", C.PURPLE))

        self._check_llm()
        stats = self.memory.get_stats()
        mem_lbl = "Memory" if lang == "en" else "Pamiec"
        sk_lbl = "Skills" if lang == "en" else "Umiejetnosci"
        print(colored(f"  {mem_lbl}: {stats['short_term_count']}ST + {stats['long_term_count']}LT | {sk_lbl}: {len(self.skills.list_names())} | Model: {self.llm.model} | Lang: {lang}", C.DIM))

        if self.llm_connected:
            self.memory.set_llm(self.reflection_llm)
            self.reflection.on_thought = self._on_background_thought
            self.reflection.start_background(llm_client=self.reflection_llm, interval=30)
            t2 = colored("ACTIVE" if lang == "en" else "AKTYWNY", C.GREEN)
        else:
            t2 = colored("OFFLINE", C.RED)

        print(colored(f"  Thread 2: {t2}", C.DIM))
        hlp = "/help = commands | /thread2 = thoughts" if lang == "en" else "/help = komendy | /thread2 = podglad mysli"
        print(colored(f"  {hlp}\n", C.DIM))
        self._update_system_prompt()

        while True:
            try:
                self._flush_thoughts()
                user_input = input(colored("you > " if lang == "en" else "ty > ", C.GREEN)).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in ("/quit", "/exit", "exit", "quit"):
                self.shutdown()
                break
            self._handle_input(user_input)

    def _on_background_thought(self, thought: str):
        self._thought_count += 1
        with self._thought_lock:
            self._thought_buffer.append(thought)
            if len(self._thought_buffer) > 50:
                self._thought_buffer = self._thought_buffer[-30:]
        if self.show_reflection:
            sys.stdout.write(colored(f"\r  [T2] {thought[:100]}\n", C.PURPLE))
            prompt_str = "you > " if get_lang() == "en" else "ty > "
            sys.stdout.write(colored(prompt_str, C.GREEN))
            sys.stdout.flush()

    def _flush_thoughts(self):
        if self.show_reflection:
            return
        with self._thought_lock:
            count = len(self._thought_buffer)
            if count == 0:
                return
            lang = get_lang()
            new_skills = sum(1 for t in self._thought_buffer if "Stworzono" in t or "Created" in t or "NAPRAWIONO" in t or "FIXED" in t)
            msg = f"  [T2] {count} new thoughts" if lang == "en" else f"  [T2] {count} nowych mysli"
            if new_skills:
                msg += f" | {new_skills} new skills" if lang == "en" else f" | {new_skills} nowych umiejetnosci"
            print(colored(msg + " (/thread2)", C.DIM))
            self._thought_buffer.clear()

    def _handle_input(self, user_input: str):
        self.interaction_count += 1
        cat = self._categorize(user_input)
        imp = self._assess_importance(user_input)
        self.memory.add(user_input, category=cat, importance=imp,
                        metadata={"type": "user_input", "n": self.interaction_count})
        if user_input.startswith("/"):
            self._handle_command(user_input)
        else:
            self._handle_conversation(user_input)

    def _handle_command(self, cmd: str):
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        lang = get_lang()

        if command == "/help":
            ttl = "=== ARIA Commands ===" if lang == "en" else "=== Komendy ARIA ==="
            print(colored(f"\n  {ttl}", C.BOLD))
            for c, d in self._cmds.items():
                print(colored(f"  {c:<24}", C.CYAN) + colored(d, C.DIM))
            print()
        elif command == "/status":
            self.print_status()
        elif command == "/memory":
            self.print_memory_dump()
        elif command == "/skills":
            self.print_skills()
        elif command == "/thread2":
            self._cmd_thread2(arg)
        elif command == "/reflect":
            if self.llm_connected:
                for t in self.reflection.reflect(trigger="manual"):
                    print(colored(f"  [T2] {t}", C.PURPLE))
            else:
                print(colored("  LLM unavailable" if lang == "en" else "  LLM niedostepne", C.RED))
        elif command == "/exec":
            if arg:
                r = self.computer.execute(arg)
                if r.get("stdout"): print(colored(r["stdout"], C.DIM))
                if r.get("stderr"): print(colored(r["stderr"], C.RED))
            else:
                print(f"  {self._ui['usage']}/exec <cmd>")
        elif command == "/python":
            if arg:
                r = self.computer.run_python(arg)
                print(colored(r.get("stdout", "") or r.get("error", "") or r.get("stderr", ""), C.DIM))
            else:
                print(f"  {self._ui['usage']}/python <code>")
        elif command == "/models":
            for m in self.llm.list_models():
                marker = " <-" if m == self.llm.model else ""
                print(colored(f"  * {m}{marker}", C.CYAN))
        elif command == "/model":
            if arg:
                old = self.llm.model
                self.llm.model = arg; self.reflection_llm.model = arg; self.llm.clear_history()
                print(colored(f"  Model: {old} -> {arg}", C.GREEN))
            else:
                print(f"  Active: {self.llm.model}")
        elif command == "/ollama":
            s = self.llm.check_connection()
            if s["connected"]:
                print(colored(f"  Ollama OK: {s['current_model']}", C.GREEN))
            else:
                print(colored(f"  No connection to {s['url']}" if lang == "en" else f"  Brak polaczenia z {s['url']}", C.RED))
        elif command == "/sysinfo":
            for k, v in self.computer.system_info().items():
                print(colored(f"  {k}: {v}", C.DIM))
        elif command == "/compress":
            r = self.memory.compress()
            print(colored(self._ui["compressed"].format(n=r["compressed"], b=r["blocks_created"]), C.CYAN))
        elif command == "/selfmodel":
            for k, v in self.reflection.self_model.items():
                if isinstance(v, list): v = ", ".join(str(x) for x in v[:5])
                print(colored(f"  {k}: {v}", C.DIM))
        else:
            print(colored(f"  {self._ui['unknown_cmd'].format(cmd=command)}", C.RED))

    def _handle_conversation(self, user_input: str):
        relevant = self.skills.find_relevant(user_input)
        skill_output = ""
        if relevant:
            skill = relevant[0]
            scripts = [s for s in skill.get_scripts() if s.suffix == ".py"]
            if scripts:
                print(colored(f"  [skill] {skill.name}/{scripts[0].name}", C.YELLOW))
                result = skill.run_script(scripts[0].name)
                if result.get("returncode", 1) == 0 and result.get("stdout"):
                    skill_output = result["stdout"][:1500]

        if not self.llm_connected:
            if skill_output:
                print(colored(f"\n{skill_output}\n", C.GREEN))
            else:
                print(colored(f"  {self._ui['offline']}\n", C.RED))
            return

        self._update_system_prompt()
        enriched = user_input
        if skill_output:
            enriched = f"{user_input}\n\n[Skill \"{relevant[0].name}\" result]:\n```\n{skill_output}\n```"

        if self.stream:
            full = ""
            print(colored("  ", C.PURPLE), end="", flush=True)
            for token in self.llm.chat_stream(enriched):
                print(token, end="", flush=True)
                full += token
            print("\n")
        else:
            full = self.llm.chat(enriched)
            print(colored(f"  {full}\n", C.PURPLE))

        full = re.sub(r'<think>.*?</think>\s*', '', full, flags=re.DOTALL).strip()
        self.memory.add(f"[Response to: {user_input[:60]}] {full[:140]}",
                        category="agent_response", importance=0.3,
                        metadata={"type": "agent_response", "n": self.interaction_count})

    def _cmd_thread2(self, arg):
        lang = get_lang()
        running = self.reflection.is_running
        cycles = self.reflection._cycle_count
        ttl = "=== Thread 2 ==="
        st = ("ACTIVE" if lang == "en" else "AKTYWNY") if running else ("STOPPED" if lang == "en" else "STOP")
        print(colored(f"\n  {ttl}", C.BOLD))
        print(colored(f"  Status: {st} | Cycles: {cycles}", C.GREEN if running else C.RED))
        n = int(arg) if arg.isdigit() else 5
        for ref in self.reflection.get_last_reflections(n):
            ts = time.strftime("%H:%M:%S", time.localtime(ref["timestamp"]))
            print(colored(f"\n  Cycle {ref.get('cycle','?')} | {ref.get('phase','?')} | {ts}", C.PURPLE))
            for t in ref.get("thoughts", [])[:5]:
                print(colored(f"    {t[:100]}", C.DIM))
        with self._thought_lock:
            if self._thought_buffer:
                lbl = "Buffer" if lang == "en" else "Bufor"
                print(colored(f"\n  {lbl} ({len(self._thought_buffer)}):", C.YELLOW))
                for t in self._thought_buffer[-10:]:
                    print(colored(f"    {t[:100]}", C.DIM))
                self._thought_buffer.clear()
        print()

    def print_status(self):
        lang = get_lang()
        stats = self.memory.get_stats()
        model = self.reflection.self_model
        print(colored(f"\n  ARIA v{model['version']} | Model: {self.llm.model} | Lang: {lang}", C.PURPLE))
        ml = "Memory" if lang == "en" else "Pamiec"
        sl = "Skills" if lang == "en" else "Umiejetnosci"
        il = "Interactions" if lang == "en" else "Interakcje"
        print(colored(f"  {ml} ST/LT/EP: {stats['short_term_count']}/{stats['long_term_count']}/{stats['episodic_count']}", C.GREEN))
        print(colored(f"  {sl}: {len(self.skills.list_names())} | {il}: {self.interaction_count}", C.DIM))
        t2s = "active" if self.reflection.is_running else "stopped"
        print(colored(f"  Thread 2: {t2s} | Cycles: {self.reflection._cycle_count}\n", C.DIM))

    def print_skills(self):
        lang = get_lang()
        skills = self.skills.list_all()
        lbl = "Skills" if lang == "en" else "Umiejetnosci"
        print(colored(f"\n  === {lbl} ({len(skills)}) ===", C.BOLD))
        if not skills:
            print(colored(f"  {self._ui['skills_empty']}", C.DIM))
        for s in skills:
            print(colored(f"  * {s['name']}", C.CYAN) + colored(f" [{s['use_count']}x] -- {s['description'][:60]}", C.DIM))
        print()

    def print_memory_dump(self):
        stats = self.memory.get_stats()
        lbl = self._ui["memory_title"]
        print(colored(f"\n  {lbl} ST:{stats['short_term_count']} LT:{stats['long_term_count']}", C.BOLD))
        for e in self.memory.short_term[-8:]:
            print(colored(f"  [{e.category}] {'*' * int(e.importance * 5)} {e.content[:80]}", C.DIM))
        print()

    def _check_llm(self):
        lang = get_lang()
        lbl = "Checking Ollama..." if lang == "en" else "Sprawdzam Ollama..."
        print(colored(f"  {lbl}", C.DIM), end=" ")
        status = self.llm.check_connection()
        if status["connected"]:
            self.llm_connected = True
            avail = "[ok]" if status["model_available"] else "[model not found]"
            print(colored(f"OK! Model: {self.llm.model} {avail}", C.GREEN))
        else:
            self.llm_connected = False
            print(colored(f"NO CONNECTION to {status['url']}", C.RED))

    def _update_system_prompt(self):
        lang = get_lang()
        template = SYSTEM_PROMPT_TEMPLATE.get(lang, SYSTEM_PROMPT_TEMPLATE["en"])
        model = self.reflection.self_model
        prompt = template.format(
            version=model.get("version", "1.0.0"),
            personality=model.get("personality", "analytical"),
            goals=", ".join(model.get("goals", [])[:3]),
            skills_section=self.skills.get_skills_prompt_section(),
            memory_context=self.memory.get_context_summary(),
        )
        self.llm.set_system_prompt(prompt)

    def _categorize(self, text):
        t = text.lower()
        for cat, kws in {
            "memory": ["pamiec", "zapamietaj", "memory", "remember", "recall"],
            "skill": ["umiejetnosc", "skill", "naucz", "teach"],
            "code": ["kod", "python", "script", "code"],
            "file": ["plik", "file", "zapisz", "save"],
            "system": ["system", "info", "status"],
        }.items():
            if any(kw in t for kw in kws):
                return cat
        return "general"

    def _assess_importance(self, text):
        t = text.lower()
        if any(w in t for w in ["zapamietaj", "wazne", "remember", "important"]):
            return 0.9
        if t in ("test", "hej", "czesc", "hello", "ok", "hi"):
            return 0.2
        if t.startswith("/"):
            return 0.4
        return 0.5

    def shutdown(self):
        lang = get_lang()
        lbl = "Stopping Thread 2..." if lang == "en" else "Zatrzymuje Thread 2..."
        print(colored(f"\n  {lbl}", C.DIM))
        self.reflection.stop_background()
        self.memory.save()
        self.config.save_self_model(self.reflection.self_model)
        bye = "Memory saved. See you!" if lang == "en" else "Pamiec zapisana. Do zobaczenia!"
        print(colored(f"  {bye}\n", C.GREEN))