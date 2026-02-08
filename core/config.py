"""Agent configuration -- loads from config.json + env vars + sensible defaults."""

import os
import json
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).parent.parent
    SKILLS_DIR = BASE_DIR / "skills"
    MEMORY_DIR = BASE_DIR / "memory"
    MEMORY_FILE = MEMORY_DIR / "memory.json"
    REFLECTION_LOG = MEMORY_DIR / "reflections.jsonl"
    SELF_MODEL_FILE = MEMORY_DIR / "self_model.json"
    CONFIG_FILE = BASE_DIR / "config.json"

    SHORT_TERM_LIMIT = 20
    COMPRESSION_RATIO = 0.5
    MAX_LONG_TERM = 100
    REFLECTION_INTERVAL = 3

    AGENT_NAME = "ARIA"
    AGENT_VERSION = "1.0.0"
    AGENT_DESCRIPTION = "Autonomous Reflective Intelligence Agent"

    LLM_CONFIG = {
        "base_url": "http://host.docker.internal:11434",
        "model": "codegemma:latest",
        "temperature": 0.7,
        "max_tokens": 4096,
        "context_length": "auto",   # "auto" = detect from Ollama API, or int
        "api_type": "ollama",
        "timeout": 120,
    }

    REFLECTION_LLM_CONFIG = {
        "base_url": "http://host.docker.internal:11434",
        "model": "codegemma:latest",
        "temperature": 0.9,
        "max_tokens": 2048,
        "context_length": "auto",
        "api_type": "ollama",
        "timeout": 60,
    }

    STREAM_RESPONSES = True

    # Language: "pl" or "en". Env var ARIA_LANG takes priority.
    LANGUAGE = "pl"

    def __init__(self):
        self.SKILLS_DIR.mkdir(parents=True, exist_ok=True)
        self.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self._load_config_file()
        # Env var overrides config.json
        env_lang = os.environ.get("ARIA_LANG", "").lower().strip()
        if env_lang in ("pl", "en"):
            self.LANGUAGE = env_lang
        # Also set it in os.environ so prompts.get_lang() can read it
        os.environ["ARIA_LANG"] = self.LANGUAGE

    def _load_config_file(self):
        if not self.CONFIG_FILE.exists():
            self._create_default_config()
            return
        try:
            data = json.loads(self.CONFIG_FILE.read_text())
            if "llm" in data:
                self.LLM_CONFIG.update(data["llm"])
            if "reflection_llm" in data:
                self.REFLECTION_LLM_CONFIG.update(data["reflection_llm"])
            if "agent" in data:
                agent = data["agent"]
                self.REFLECTION_INTERVAL = agent.get("reflection_interval", self.REFLECTION_INTERVAL)
                self.STREAM_RESPONSES = agent.get("stream_responses", self.STREAM_RESPONSES)
                self.SHORT_TERM_LIMIT = agent.get("short_term_limit", self.SHORT_TERM_LIMIT)
                lang = agent.get("language", "").lower().strip()
                if lang in ("pl", "en"):
                    self.LANGUAGE = lang
        except (json.JSONDecodeError, KeyError):
            pass

    def _create_default_config(self):
        default = {
            "llm": self.LLM_CONFIG,
            "reflection_llm": self.REFLECTION_LLM_CONFIG,
            "agent": {
                "reflection_interval": self.REFLECTION_INTERVAL,
                "stream_responses": self.STREAM_RESPONSES,
                "short_term_limit": self.SHORT_TERM_LIMIT,
                "language": self.LANGUAGE,
            },
        }
        self.CONFIG_FILE.write_text(json.dumps(default, ensure_ascii=False, indent=4))

    def load_self_model(self) -> dict:
        if self.SELF_MODEL_FILE.exists():
            return json.loads(self.SELF_MODEL_FILE.read_text())
        from core.prompts import DEFAULT_SELF_MODEL
        lang_data = DEFAULT_SELF_MODEL.get(self.LANGUAGE, DEFAULT_SELF_MODEL["en"])
        return {
            "version": self.AGENT_VERSION,
            "personality": lang_data["personality"],
            "strengths": [],
            "weaknesses": [],
            "goals": lang_data["goals"],
            "improvements_applied": 0,
            "total_interactions": 0,
            "skills_created": 0,
        }

    def save_self_model(self, model: dict):
        self.SELF_MODEL_FILE.write_text(json.dumps(model, ensure_ascii=False, indent=2))