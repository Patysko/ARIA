"""
Central logging for ARIA agent.

All events -- user messages, agent responses, CoT steps, Thread 2 thoughts,
skill executions, errors -- go to a single rotating log file.

Usage:
    from core.logger import log
    log.user("Hello ARIA")
    log.agent("Response text")
    log.cot("analyze", {"intent": "question"})
    log.thread2("skill_building", "Creating new skill")
    log.skill("system-monitor", "main.py", 0, "output...")
    log.error("something broke", exc_info=True)
"""

import logging
import time
import json
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler


class AriaLogger:
    """Structured logger that writes all events to aria.log."""

    def __init__(self):
        self._logger = logging.getLogger("aria")
        self._logger.setLevel(logging.DEBUG)
        self._initialized = False

    def init(self, log_dir: Path):
        """Initialize file handler. Called once from Config."""
        if self._initialized:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "aria.log"

        handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=3, encoding="utf-8",
        )
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)

        # Also log to stderr at WARNING+ level
        stderr = logging.StreamHandler()
        stderr.setLevel(logging.WARNING)
        stderr.setFormatter(formatter)
        self._logger.addHandler(stderr)

        self._initialized = True
        self._logger.info("=== ARIA session started ===")

    def _json(self, data: dict) -> str:
        """Compact JSON for structured log entries."""
        return json.dumps(data, ensure_ascii=False, default=str)

    # --- High-level log methods ---

    def user(self, message: str, interaction_n: int = 0):
        self._logger.info(f"USER [{interaction_n}] | {message[:500]}")

    def agent(self, reply: str, interaction_n: int = 0,
              skills_used: list = None):
        meta = ""
        if skills_used:
            meta = f" skills={skills_used}"
        self._logger.info(f"AGENT [{interaction_n}]{meta} | {reply[:500]}")

    def cot(self, step: str, data: dict):
        self._logger.debug(f"COT/{step} | {self._json(data)}")

    def thread2(self, phase: str, message: str):
        self._logger.info(f"T2/{phase} | {message[:500]}")

    def thread2_to_user(self, message: str):
        self._logger.info(f"T2->USER | {message[:500]}")

    def skill(self, skill_name: str, script: str,
              returncode: int, output: str = ""):
        self._logger.info(
            f"SKILL {skill_name}/{script} exit={returncode} | {output[:300]}"
        )

    def skill_error(self, skill_name: str, script: str, error: str):
        self._logger.warning(f"SKILL_ERR {skill_name}/{script} | {error[:500]}")

    def task(self, action: str, task_id: str, details: str = ""):
        self._logger.info(f"TASK/{action} {task_id} | {details[:300]}")

    def error(self, message: str, exc_info: bool = False):
        self._logger.error(f"ERROR | {message}", exc_info=exc_info)

    def warning(self, message: str):
        self._logger.warning(f"WARN | {message}")

    def info(self, message: str):
        self._logger.info(message)

    def debug(self, message: str):
        self._logger.debug(message)

    def command(self, cmd: str):
        self._logger.info(f"CMD | {cmd}")


# Singleton
log = AriaLogger()