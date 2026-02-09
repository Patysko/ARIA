"""
Reflection Thread for ARIA -- TRUE BACKGROUND THREAD.
All strings via core.prompts for PL/EN support.

Phases:
- introspection, pattern_analysis, skill_planning, skill_building,
  skill_testing, self_improvement, knowledge_synthesis, exploration,
  task_processing, proactive_messaging
"""

import json
import re
import time
import threading
from pathlib import Path
from collections import Counter
from typing import Optional

from core.computer import ComputerTools
from core.logger import log
from core.prompts import (
    get_lang, T2_SYSTEM, T2_BUILD, T2_FIX, T2_THINK,
    T2_PHASE_INSTRUCTIONS, T2_MSG, T2_TASK_CHECK, T2_PROACTIVE,
    T2_EXPLORE_PLAN, T2_SAFETY_CHECK,
)

PHASES = [
    "introspection", "pattern_analysis", "skill_planning", "skill_building",
    "skill_testing", "self_improvement", "knowledge_synthesis", "exploration",
    "task_processing", "system_exploration",
]

MAX_FIX_ATTEMPTS = 2

BLOCKED_PACKAGES = {
    "torch", "pytorch", "tensorflow", "tf", "jax", "flax",
    "transformers", "diffusers", "accelerate", "bitsandbytes",
    "cuda", "cudnn", "tensorrt", "onnxruntime-gpu",
    "detectron2", "mmdet", "mmcv", "paddlepaddle",
}


class ReflectionThread:
    def __init__(self, config, memory, skills_manager):
        self.config = config
        self.memory = memory
        self.skills = skills_manager
        self.self_model = config.load_self_model()
        self.computer = ComputerTools()
        self.reflections: list[dict] = []
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._llm = None
        self._phase_index = 0
        self._cycle_count = 0
        self._lock = threading.Lock()
        self.on_thought: Optional[callable] = None
        self.on_user_message: Optional[callable] = None
        self.interval = 30
        self._fix_attempts: dict[str, int] = {}
        self._load_log()

    @property
    def _m(self):
        return T2_MSG.get(get_lang(), T2_MSG["en"])

    def _load_log(self):
        lp = self.config.REFLECTION_LOG
        if lp.exists():
            try:
                for line in lp.read_text(errors="replace").strip().split("\n"):
                    if line.strip():
                        self.reflections.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                pass
        if self.reflections:
            p = self.reflections[-1].get("phase", "introspection")
            if p in PHASES:
                self._phase_index = (PHASES.index(p) + 1) % len(PHASES)

    def _save_reflection(self, r: dict):
        with self._lock:
            self.reflections.append(r)
        try:
            with open(self.config.REFLECTION_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # --- Background thread ---

    def start_background(self, llm_client, interval=30):
        self._llm = llm_client
        self.interval = interval
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ARIA-T2")
        self._thread.start()

    def stop_background(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    @property
    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _loop(self):
        time.sleep(5)
        while not self._stop_event.is_set():
            try:
                self._run_cycle()
            except Exception as e:
                self._emit(self._m["llm_error"].format(e=e))
            for _ in range(self.interval * 2):
                if self._stop_event.is_set():
                    return
                time.sleep(0.5)

    def _run_cycle(self):
        self._cycle_count += 1
        phase = PHASES[self._phase_index]
        self._phase_index = (self._phase_index + 1) % len(PHASES)
        self._emit(self._m["cycle"].format(n=self._cycle_count, phase=phase))
        log.thread2(phase, f"Cycle {self._cycle_count} starting")

        if not self._llm:
            thoughts = self._rule_based(phase)
        elif phase == "skill_building":
            thoughts = self._phase_build()
        elif phase == "skill_testing":
            thoughts = self._phase_test()
        elif phase == "task_processing":
            thoughts = self._phase_tasks()
        elif phase == "system_exploration":
            thoughts = self._phase_explore_system()
        else:
            thoughts = self._phase_think(phase)

        # Proactive messaging every 5 cycles
        if self._cycle_count % 5 == 0:
            self._try_proactive()

        self._update_self_model()
        self._save_reflection({
            "timestamp": time.time(), "cycle": self._cycle_count,
            "phase": phase, "thoughts": thoughts,
            "stats": self.memory.get_stats(),
            "skills_count": len(self.skills.list_names()),
        })
        log.thread2(phase, f"Cycle {self._cycle_count} done, {len(thoughts)} thoughts")

    # --- Skill building ---

    def _phase_build(self) -> list[str]:
        m = self._m
        lang = get_lang()
        thoughts = []
        ctx = self._build_context()

        # Build skills detail with actual code snippets
        skills_detail = ""
        for sname, skill in list(self.skills.skills.items()):
            scripts = skill.get_scripts()
            for sp in scripts:
                if sp.suffix == ".py":
                    try:
                        code = sp.read_text(errors="replace")[:500]
                        skills_detail += f"\n### {sname}/{sp.name}:\n```python\n{code}\n```\n"
                    except Exception:
                        pass

        self._emit(m["designing"])
        try:
            self._llm.set_system_prompt(T2_SYSTEM[lang])
            response = self._llm.chat(
                T2_BUILD[lang].format(
                    skills_list=ctx["skills_list"],
                    skills_detail=skills_detail or "(no code)",
                    recent_interactions=ctx["recent"],
                    previous_thoughts=ctx["prev_thoughts"],
                ),
                include_history=False,
            )
        except Exception as e:
            return [m["llm_error"].format(e=e)]

        sd = self._extract_json(response) or self._extract_skill_from_text(response)
        if not sd or not sd.get("name"):
            self._emit(m["bad_json"])
            return [m["bad_json"]]

        action = sd.get("action", "create")
        name = sd["name"]
        code = self._clean_code(sd.get("script_code", ""))
        if not code.strip():
            return [m["no_code"].format(name=name)]

        # Similarity check: reject if too similar to existing skill
        if action == "create" and self._is_duplicate_skill(name, sd.get("description", "")):
            self._emit(m["exists"].format(name=name))
            return [m["exists"].format(name=name)]

        # Handle IMPROVE action
        if action == "improve" and name in self.skills.list_names():
            return self._improve_skill(name, sd, code, thoughts)

        # Handle CREATE action
        if name in self.skills.list_names():
            self._emit(m["exists"].format(name=name))
            return [m["exists"].format(name=name)]

        pip_pkgs = self._filter_packages(sd.get("pip_packages", []))
        detected = self._filter_packages(self.computer.extract_imports(code))
        all_deps = list(set(pip_pkgs + detected))
        if all_deps:
            self._emit(m["installing"].format(deps=", ".join(all_deps)))
            r = self.computer.pip_install(all_deps)
            thoughts.append(m["installed"].format(deps=", ".join(all_deps)) if r["success"]
                           else m["pip_error"].format(err=r.get("error", "")[:150]))

        self._emit(m["creating"].format(name=name))
        try:
            skill = self.skills.create_skill(
                name=name, description=sd.get("description", "Auto-skill"),
                instructions=sd.get("instructions", ""),
                scripts={sd.get("script_name", "main.py"): code},
            )
        except Exception as e:
            return [m["create_error"].format(e=e)]

        thoughts.append(m["created"].format(name=name))
        sn = sd.get("script_name", "main.py")
        test = self._test_skill(skill, sn)
        thoughts.append(test["message"])
        if not test["success"]:
            ta = sd.get("test_args", [])
            if ta:
                t2 = self._test_skill(skill, sn, test_args=ta)
                thoughts.append(t2["message"])
                if t2["success"]:
                    test = t2
        if not test["success"]:
            thoughts.append(self._fix_skill(skill, sn, code, test["error"]))

        self.memory.add(
            f"Thread 2 created skill: {name} - {sd.get('description','')}",
            category="skill-creation", importance=0.9,
            metadata={"source": "thread2", "skill": name},
        )
        self.memory.add_episodic(f"Created skill: {name}", "Thread 2")
        self._emit_to_user(m["proactive_skill"].format(name=name, desc=sd.get("description", "")))
        log.thread2("skill_building", f"Created skill: {name}")

        self.self_model["skills_created"] = len(self.skills.list_names())
        self.self_model["improvements_applied"] = self.self_model.get("improvements_applied", 0) + 1
        self.config.save_self_model(self.self_model)
        return thoughts

    def _improve_skill(self, name: str, sd: dict, code: str, thoughts: list) -> list[str]:
        """Improve an existing skill with new code."""
        m = self._m
        skill = self.skills.get(name)
        if not skill:
            return [m["exists"].format(name=name)]

        self._emit(m["improving"].format(name=name))
        sn = sd.get("script_name", "main.py")
        script_path = skill.scripts_dir / sn

        # Install deps
        pip_pkgs = self._filter_packages(sd.get("pip_packages", []))
        detected = self._filter_packages(self.computer.extract_imports(code))
        all_deps = list(set(pip_pkgs + detected))
        if all_deps:
            self.computer.pip_install(all_deps)

        # Backup old code, write new
        old_code = ""
        if script_path.exists():
            old_code = script_path.read_text(errors="replace")
        script_path.write_text(code)
        script_path.chmod(0o755)

        # Update SKILL.md if description changed
        if sd.get("description"):
            skill.description = sd["description"]
        if sd.get("instructions"):
            skill.instructions = sd["instructions"]

        # Test
        test = self._test_skill(skill, sn)
        thoughts.append(test["message"])

        if not test["success"] and old_code:
            # Rollback on failure
            script_path.write_text(old_code)
            thoughts.append(f"Rolled back {name} — new code failed")
            return thoughts

        reason = sd.get("reason", "improved")
        self._emit_to_user(m["improved"].format(name=name, reason=reason))
        self.memory.add(f"Thread 2 improved skill: {name} — {reason}",
                       category="skill-creation", importance=0.8,
                       metadata={"source": "thread2", "skill": name})
        log.thread2("skill_building", f"Improved skill: {name}")
        return thoughts

    def _is_duplicate_skill(self, new_name: str, new_desc: str) -> bool:
        """Check if a proposed skill is too similar to existing ones."""
        new_words = set(new_name.replace("-", " ").split() +
                       new_desc.lower().split())
        for existing in self.skills.skills.values():
            existing_words = set(
                existing.name.replace("-", " ").split() +
                existing.description.lower().split()
            )
            overlap = len(new_words & existing_words)
            total = len(new_words | existing_words)
            if total > 0 and overlap / total > 0.5:
                return True
        return False

    # --- Skill testing ---

    def _phase_test(self) -> list[str]:
        m = self._m
        thoughts = []
        tested = fixed = 0
        self._fix_attempts.clear()

        for sn, skill in list(self.skills.skills.items()):
            for sp in skill.get_scripts():
                if sp.suffix == ".py":
                    tested += 1
                    r = self._test_skill(skill, sp.name)
                    if not r["success"]:
                        args = self._infer_test_args(skill)
                        if args:
                            r2 = self._test_skill(skill, sp.name, test_args=args)
                            if r2["success"]:
                                continue
                        fk = f"{sn}/{sp.name}"
                        att = self._fix_attempts.get(fk, 0)
                        if att < MAX_FIX_ATTEMPTS:
                            self._emit(m["fixing"].format(key=fk, n=att + 1, max=MAX_FIX_ATTEMPTS))
                            fix_result = self._fix_skill(skill, sp.name, sp.read_text(errors="replace"), r["error"])
                            thoughts.append(fix_result)
                            self._fix_attempts[fk] = att + 1
                            fixed += 1
                            # Clear error log if fix succeeded
                            if "NAPRAWIONO" in fix_result or "FIXED" in fix_result:
                                skill.clear_errors()
                                self._emit(m["errors_cleared"].format(name=sn))
                                log.thread2("skill_testing", f"Cleared error log: {sn}")
                        else:
                            self._emit(m["fix_skip"].format(key=fk))
                    else:
                        self._emit(m["test_ok"].format(path=f"{sn}/{sp.name}", args=""))

        msg = m["test_summary"].format(tested=tested, fixed=fixed)
        thoughts.append(msg)
        self._emit(f"[T2] {msg}")
        log.thread2("skill_testing", msg)
        return thoughts

    # --- Helpers ---

    def _filter_packages(self, pkgs):
        m = self._m
        out = []
        for p in pkgs:
            if p.lower().strip() in BLOCKED_PACKAGES:
                self._emit(m["blocked_pkg"].format(pkg=p))
                continue
            out.append(p)
        return out

    def _infer_test_args(self, skill):
        try:
            inst = skill.instructions or ""
            r = re.findall(r'(?:python3?\s+\S+\.py|scripts/\S+\.py)\s+(.+?)(?:\n|$)', inst)
            if r:
                ex = re.sub(r'<\w+[\w_]*>', '.', r[0].strip())
                ex = re.sub(r'\[([^\]]*)\]', '', ex)
                return [a.strip() for a in ex.split() if a.strip()] or []
            if re.findall(r'sys\.argv\[(\d+)\]', inst):
                return ["."]
        except Exception:
            pass
        return []

    def _test_skill(self, skill, script_name, _retry=False, test_args=None):
        m = self._m
        ad = f" (args: {test_args})" if test_args else ""
        self._emit(m["testing"].format(path=f"{skill.name}/{script_name}", args=ad))
        result = skill.run_script(script_name, args=test_args)

        if "error" in result:
            msg = m["test_fail"].format(path=f"{skill.name}/{script_name}", err=result["error"])
            self._emit(msg)
            return {"success": False, "error": result["error"], "message": msg}

        if result.get("returncode", 1) != 0:
            error = result.get("stderr", "") or result.get("stdout", "")
            if not _retry and "ModuleNotFoundError" in error:
                missing = self._extract_missing_module(error)
                if missing:
                    if missing.lower() in BLOCKED_PACKAGES:
                        msg = m["test_heavy"].format(name=skill.name, pkg=missing)
                        self._emit(msg)
                        return {"success": False, "error": msg, "message": msg}
                    self._emit(m["missing_mod"].format(mod=missing))
                    if self.computer.pip_install([missing])["success"]:
                        return self._test_skill(skill, script_name, _retry=True, test_args=test_args)
            msg = m["test_fail"].format(path=f"{skill.name}/{script_name}", err=f"exit={result['returncode']} {error[:150]}")
            self._emit(msg)
            return {"success": False, "error": error[:500], "message": msg}

        msg = m["test_ok"].format(path=f"{skill.name}/{script_name}", args=ad)
        self._emit(msg)
        return {"success": True, "output": result.get("stdout", "")[:200], "message": msg}

    def _extract_missing_module(self, error):
        r = re.search(r"No module named ['\"](\w+)['\"]", error)
        if r:
            n = r.group(1)
            pm = {'cv2': 'opencv-python', 'PIL': 'Pillow', 'bs4': 'beautifulsoup4',
                   'sklearn': 'scikit-learn', 'yaml': 'pyyaml', 'dotenv': 'python-dotenv',
                   'dateutil': 'python-dateutil'}
            return pm.get(n, n)
        return ""

    def _fix_skill(self, skill, script_name, code, error):
        m = self._m
        lang = get_lang()
        if not self._llm:
            return m["fix_no_llm"]
        pip_list = self.computer.execute("pip3 list --format=columns 2>/dev/null | head -30")
        installed = pip_list.get("stdout", "")[:500] if pip_list.get("success") else "unknown"
        try:
            self._llm.set_system_prompt(T2_SYSTEM[lang])
            response = self._llm.chat(T2_FIX[lang].format(
                name=skill.name, error=error[:500], code=code[:2000], installed_packages=installed
            ), include_history=False)
            fixed = self._clean_code(response)
            if not fixed.strip():
                return m["fix_no_code"].format(name=skill.name)

            pip_deps = []
            for line in fixed.split("\n")[:3]:
                if line.strip().upper().startswith("# PIP:"):
                    pip_deps = self._filter_packages(line.split(":", 1)[1].strip().split())
                    break
            detected = self._filter_packages(self.computer.extract_imports(fixed))
            if list(set(pip_deps + detected)):
                self.computer.pip_install(list(set(pip_deps + detected)))

            sp = skill.scripts_dir / script_name
            sp.write_text(fixed, encoding="utf-8")
            sp.chmod(0o755)

            r = self._test_skill(skill, script_name)
            if r["success"]:
                return m["fix_ok"].format(path=f"{skill.name}/{script_name}")
            args = self._infer_test_args(skill)
            if args:
                r2 = self._test_skill(skill, script_name, test_args=args)
                if r2["success"]:
                    return m["fix_ok"].format(path=f"{skill.name}/{script_name}")
            return m["fix_fail"].format(err=r["error"][:100])
        except Exception as e:
            return m["fix_error"].format(e=e)

    # --- Generic phase ---

    def _phase_think(self, phase):
        m = self._m
        lang = get_lang()
        ctx = self._build_context()
        phase_inst = T2_PHASE_INSTRUCTIONS.get(lang, T2_PHASE_INSTRUCTIONS["en"])
        try:
            self._llm.set_system_prompt(T2_SYSTEM[lang])
            response = self._llm.chat(T2_THINK[lang].format(
                version=self.self_model.get("version", "1.0.0"),
                skills_list=ctx["skills_list"],
                memory_stats=ctx["memory_stats"],
                recent_interactions=ctx["recent"],
                previous_thoughts=ctx["prev_thoughts"],
                phase=phase,
                phase_instruction=phase_inst.get(phase, "Think freely."),
            ), include_history=False)
            clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip() or response
            thoughts = []
            for line in clean.split("\n"):
                line = line.strip()
                if line and len(line) > 5:
                    thoughts.append(line)
                    self._emit(f"[T2] {line[:150]}")
            if phase == "skill_planning" and thoughts:
                kws = m.get("proposal_keywords", [])
                proposals = [t for t in thoughts if any(w in t.lower() for w in kws)]
                if proposals:
                    self._emit_to_user(m["proactive_proposals"].format(
                        list="\n".join(f"- {p[:120]}" for p in proposals[:3])
                    ))
            self.memory.add(f"[T2/{phase}] {clean[:150]}", category="self-reflection", importance=0.4,
                           metadata={"phase": phase, "cycle": self._cycle_count, "source": "thread2"})
            return thoughts[:10]
        except Exception as e:
            return [m["llm_error"].format(e=e)]

    # --- Task processing phase ---

    def _phase_tasks(self) -> list[str]:
        """Check pending tasks and attempt to complete any that are now possible."""
        m = self._m
        thoughts = []
        pending = self.memory.get_pending_tasks()

        if not pending:
            thoughts.append("[T2] No pending tasks")
            return thoughts

        lang = get_lang()

        # Ask LLM which tasks are actionable now
        tasks_desc = "\n".join(
            f"  - id={t['id']} | msg=\"{t['message'][:100]}\" | reason={t['reason']} | attempts={t['attempts']}"
            for t in pending[:10]
        )
        skills_list = ", ".join(self.skills.list_names()) or "none"

        try:
            self._llm.set_system_prompt(T2_SYSTEM[lang])
            raw = self._llm.chat(
                T2_TASK_CHECK[lang].format(tasks=tasks_desc, skills_list=skills_list),
                include_history=False,
            )
            parsed = self._extract_json(raw)
        except Exception as e:
            log.error(f"T2 task check failed: {e}")
            return [m["llm_error"].format(e=e)]

        if not parsed:
            return ["[T2] Could not parse task check response"]

        for item in parsed.get("actionable", []):
            task_id = item.get("task_id", "")
            task = next((t for t in pending if t["id"] == task_id), None)
            if not task:
                continue
            if task["attempts"] >= 3:
                self.memory.mark_task(task_id, "failed", "max attempts")
                log.task("failed", task_id, "max attempts reached")
                continue

            self._emit(m["task_attempting"].format(message=task["message"][:80]))
            self.memory.increment_task_attempt(task_id)
            log.task("attempting", task_id, task["message"][:100])

            # Try to execute the task
            skill_name = item.get("skill")
            args = item.get("args", [])
            result_text = ""

            if skill_name and self.skills.get(skill_name):
                skill = self.skills.get(skill_name)
                scripts = [s for s in skill.get_scripts() if s.suffix == ".py"]
                if scripts:
                    r = skill.run_script(scripts[0].name, args=args)
                    if r.get("returncode", 1) == 0 and r.get("stdout"):
                        result_text = r["stdout"][:500]
                    else:
                        err = r.get("stderr", r.get("error", ""))[:200]
                        self._emit(m["task_failed"].format(reason=err))
                        thoughts.append(m["task_failed"].format(reason=err))
                        log.task("failed", task_id, err)
                        continue

            if not result_text and self._llm:
                # Try direct LLM answer
                try:
                    from core.prompts import CHAT_SYSTEM
                    self._llm.set_system_prompt(CHAT_SYSTEM[lang])
                    result_text = self._llm.chat(
                        f"Earlier the user asked: \"{task['message']}\"\n"
                        f"I couldn't help then because: {task['reason']}\n"
                        f"Now I have skills: {skills_list}\n"
                        f"Please answer the user's original question.",
                        include_history=False,
                    )
                except Exception as e:
                    log.error(f"T2 task LLM failed: {e}")
                    continue

            if result_text:
                self.memory.mark_task(task_id, "done", result_text[:300])
                # Send result to user!
                user_msg = m["task_completed"].format(
                    message=task["message"][:100],
                    result=result_text[:300],
                )
                self._emit_to_user(user_msg)
                thoughts.append(f"✅ Completed task: {task['message'][:80]}")
                log.task("completed", task_id, result_text[:200])

        # Clean up old tasks
        self.memory.cleanup_tasks(max_age_hours=48)
        thoughts.append(f"[T2] Processed {len(pending)} pending tasks")
        return thoughts

    # --- Proactive messaging phase ---

    def _phase_explore_system(self) -> list[str]:
        """Explore the system using safe read-only shell commands."""
        m = self._m
        thoughts = []
        lang = get_lang()

        self._emit(m["exploring"])

        # Gather what we already know
        sysinfo = self.computer.system_info()
        sysinfo_str = "\n".join(f"  {k}: {v}" for k, v in sysinfo.items())

        known = []
        for entry in self.memory.short_term[-20:]:
            if entry.category == "system_discovery":
                known.append(entry.content[:100])
        for block in self.memory.long_term[-10:]:
            if block.category == "system_discovery":
                known.append(block.summary[:100])
        known_str = "\n".join(known) if known else "(first exploration)"

        # Ask LLM to plan exploration
        try:
            self._llm.set_system_prompt(T2_SYSTEM[lang])
            raw = self._llm.chat(
                T2_EXPLORE_PLAN[lang].format(
                    sysinfo=sysinfo_str,
                    known_info=known_str,
                ),
                include_history=False,
            )
            plan = self._extract_json(raw)
        except Exception as e:
            return [m["llm_error"].format(e=e)]

        if not plan or not plan.get("commands"):
            return ["[T2] No exploration commands planned"]

        findings = []
        for item in plan["commands"][:5]:
            cmd = item.get("cmd", "").strip()
            purpose = item.get("purpose", "")
            if not cmd:
                continue

            # Safety check: hardcoded blocklist first
            if self._is_obviously_unsafe(cmd):
                self._emit(m["explore_unsafe"].format(cmd=cmd))
                thoughts.append(f"Skipped unsafe: {cmd}")
                continue

            # LLM safety check
            try:
                safety_raw = self._llm.chat(
                    T2_SAFETY_CHECK[lang].format(command=cmd, purpose=purpose),
                    include_history=False,
                )
                safety = self._extract_json(safety_raw)
                if not safety or not safety.get("safe"):
                    reason = safety.get("reason", "unknown") if safety else "no response"
                    self._emit(m["explore_unsafe"].format(cmd=cmd))
                    thoughts.append(f"LLM rejected: {cmd} ({reason})")
                    continue
            except Exception:
                continue

            # Execute
            self._emit(m["explore_cmd"].format(cmd=cmd, purpose=purpose))
            result = self.computer.execute(cmd, timeout=15)
            output = (result.get("stdout", "") or result.get("stderr", ""))[:300]

            if result.get("returncode", 1) == 0 and output:
                self._emit(m["explore_result"].format(result=output[:100]))
                findings.append(f"{purpose}: {output[:150]}")
                thoughts.append(f"Discovered ({purpose}): {output[:100]}")
            else:
                thoughts.append(f"Failed: {cmd}")

        # Store discoveries in memory
        if findings:
            summary = "; ".join(findings)[:400]
            self.memory.add(
                f"System exploration: {summary}",
                category="system_discovery", importance=0.6,
                metadata={"source": "thread2", "type": "exploration"},
            )
            self._emit(m["explore_summary"].format(findings=summary[:150]))
            log.thread2("system_exploration", f"Found: {summary[:200]}")

        return thoughts

    @staticmethod
    def _is_obviously_unsafe(cmd: str) -> bool:
        """Hardcoded blocklist for obviously destructive commands."""
        dangerous = [
            "rm ", "rm\t", "rmdir", "mkfs", "dd ", "format",
            "sudo", "su ", "chmod -R", "chown -R",
            "> /", ">> /", "tee /",
            "shutdown", "reboot", "halt", "poweroff",
            "kill -9", "killall", "pkill",
            ":(){", "fork", ":(){ :|:&",
        ]
        cl = cmd.lower().strip()
        return any(d in cl for d in dangerous)

    def _try_proactive(self):
        """Decide whether to proactively message the user. Called periodically."""
        m = self._m
        lang = get_lang()
        ctx = self._build_context()

        last_interaction = 0
        for entry in reversed(self.memory.short_term):
            if entry.metadata.get("type") == "user_input":
                last_interaction = entry.timestamp
                break
        idle_secs = time.time() - last_interaction if last_interaction else 9999
        idle_str = f"{int(idle_secs // 60)}m" if idle_secs < 3600 else f"{idle_secs // 3600:.1f}h"

        if idle_secs < 300:
            return

        try:
            self._llm.set_system_prompt(T2_SYSTEM[lang])
            raw = self._llm.chat(
                T2_PROACTIVE[lang].format(
                    recent=ctx["recent"],
                    skills_list=ctx["skills_list"],
                    cycle=self._cycle_count,
                    idle_time=idle_str,
                ),
                include_history=False,
            )
            parsed = self._extract_json(raw)
        except Exception:
            return

        if parsed and parsed.get("should_message") and parsed.get("message"):
            msg_text = parsed["message"]
            self._emit_to_user(m["proactive_greeting"].format(message=msg_text))
            log.thread2("proactive", f"Sent: {msg_text[:200]}")

    def _rule_based(self, phase):
        thoughts = []
        stats = self.memory.get_stats()
        if phase == "introspection":
            thoughts.append(f"State: {stats['short_term_count']} memories, {len(self.skills.list_names())} skills")
        elif phase == "pattern_analysis":
            cats = Counter(e.category for e in self.memory.short_term)
            if cats:
                top = cats.most_common(1)[0]
                thoughts.append(f"Top category: '{top[0]}' ({top[1]}x)")
        for t in thoughts:
            self._emit(f"[T2] {t}")
        return thoughts

    # --- Context (isolated from chat CoT) ---

    def _build_context(self):
        stats = self.memory.get_stats()
        recent = []
        for entry in self.memory.short_term[-20:]:
            mt = entry.metadata.get("type", "")
            src = entry.metadata.get("source", "")
            if src == "thread2":
                continue
            if mt in ("command_response", "system_internal"):
                continue
            if entry.category in ("command_output", "system_internal", "self-reflection"):
                continue
            if entry.content.startswith("/"):
                continue
            recent.append(f"  [{entry.category}] {entry.content[:100]}")
        recent = recent[-8:]
        prev = []
        with self._lock:
            for ref in self.reflections[-2:]:
                for t in ref.get("thoughts", [])[:3]:
                    prev.append(f"  - {t[:120]}")
        return {
            "skills_list": ", ".join(self.skills.list_names()) or "none",
            "memory_stats": f"ST:{stats['short_term_count']} LT:{stats['long_term_count']} EP:{stats['episodic_count']}",
            "recent": "\n".join(recent) or "  (no interactions)",
            "prev_thoughts": "\n".join(prev) or "  (first reflection)",
        }

    # --- Emit ---

    def _emit(self, msg):
        log.debug(f"T2_THOUGHT: {msg[:200]}")
        if self.on_thought:
            try:
                self.on_thought(msg)
            except Exception:
                pass

    def _emit_to_user(self, msg):
        log.thread2_to_user(msg)
        if self.on_user_message:
            try:
                self.on_user_message(msg)
            except Exception:
                pass

    def _update_self_model(self):
        m = self._m
        self.self_model["total_interactions"] = (
            self.memory.get_stats()["short_term_count"]
            + sum(b.source_count for b in self.memory.long_term)
        )
        self.self_model["skills_created"] = len(self.skills.list_names())
        if self._cycle_count > 0 and self._cycle_count % 10 == 0:
            parts = self.self_model["version"].split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            self.self_model["version"] = ".".join(parts)
            self._emit(m["new_version"].format(ver=self.self_model["version"]))
        self.config.save_self_model(self.self_model)

    # --- JSON parsing ---

    def _extract_json(self, text):
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        i = text.find("{")
        if i < 0:
            return None
        depth = 0; in_str = False; esc = False; end = i
        for j in range(i, len(text)):
            c = text[j]
            if esc: esc = False; continue
            if c == '\\': esc = True; continue
            if c == '"': in_str = not in_str; continue
            if in_str: continue
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0: end = j + 1; break
        if depth != 0: return None
        try:
            d = json.loads(text[i:end])
            return d if isinstance(d, dict) and d.get("name") else None
        except json.JSONDecodeError:
            try: return json.loads(text[i:end].replace("'", '"'))
            except: return None

    def _extract_skill_from_text(self, text):
        data = {}; code_lines = []; in_code = False
        for line in text.split("\n"):
            low = line.strip().lower()
            if low.startswith(("nazwa:", "name:")): data["name"] = line.split(":", 1)[1].strip().strip('"\'')
            elif low.startswith(("opis:", "description:")): data["description"] = line.split(":", 1)[1].strip().strip('"')
            elif "```python" in low: in_code = True
            elif in_code and "```" in line: in_code = False
            elif in_code: code_lines.append(line)
        if code_lines: data["script_code"] = "\n".join(code_lines)
        data.setdefault("script_name", "main.py")
        return data if data.get("name") else None

    def _clean_code(self, text):
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```": lines = lines[:-1]
            text = "\n".join(lines)
        return re.sub(r'\n```\s*$', '', text).strip()

    # --- Public API ---

    def reflect(self, trigger="manual"):
        phase = PHASES[self._phase_index]
        self._phase_index = (self._phase_index + 1) % len(PHASES)
        self._cycle_count += 1
        log.thread2(phase, f"Manual reflect, cycle {self._cycle_count}")
        if not self._llm: return self._rule_based(phase)
        if phase == "skill_building": return self._phase_build()
        if phase == "skill_testing": return self._phase_test()
        if phase == "task_processing": return self._phase_tasks()
        if phase == "proactive_messaging": return self._phase_proactive()
        return self._phase_think(phase)

    def get_last_reflections(self, n=3):
        with self._lock:
            return self.reflections[-n:]