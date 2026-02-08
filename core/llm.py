"""
LLM Client for ARIA — connects to Ollama or OpenAI-compatible endpoints.

Features:
- AUTO-DETECT model context length from Ollama /api/show
- Fill context window aggressively — use all available tokens
- Ollama native API + OpenAI-compatible mode
- Streaming support
- Model pull integrated into model switching
"""

import json
import sys
import urllib.request
import urllib.error
from typing import Generator, Optional


# Fallback context sizes for known model families (if API detection fails)
MODEL_CONTEXT_DEFAULTS = {
    "llama3":    8192,
    "llama3.1":  131072,
    "llama3.2":  131072,
    "llama3.3":  131072,
    "qwen":      32768,
    "qwen2":     32768,
    "qwen2.5":   32768,
    "qwen3":     32768,
    "mistral":   32768,
    "mixtral":   32768,
    "gemma":     8192,
    "gemma2":    8192,
    "codegemma": 8192,
    "phi3":      128000,
    "phi4":      16384,
    "deepseek":  65536,
    "deepseek-coder": 65536,
    "codellama": 16384,
    "command-r": 131072,
    "yi":        200000,
    "solar":     4096,
    "nous-hermes": 8192,
}


class LLMClient:
    """Connects ARIA to a local LLM via Ollama."""

    def __init__(self, config: dict):
        self.base_url = config.get("base_url", "http://host.docker.internal:11434")
        self.model = config.get("model", "codegemma:latest")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
        self.api_type = config.get("api_type", "ollama")
        self.timeout = config.get("timeout", 120)
        self.conversation_history: list[dict] = []
        self.system_prompt = ""

        # Context length: "auto" = detect from Ollama, int = fixed
        raw_ctx = config.get("context_length", "auto")
        self._context_length_config = raw_ctx
        self._detected_context_length: Optional[int] = None

        if isinstance(raw_ctx, int) and raw_ctx > 0:
            self._detected_context_length = raw_ctx

    @property
    def context_length(self) -> int:
        """Effective context length — auto-detected or configured."""
        if self._detected_context_length:
            return self._detected_context_length
        return self._guess_context_from_model_name()

    def detect_context_length(self) -> int:
        """Query Ollama /api/show to get the actual model context length.
        Called once after connection is confirmed."""
        if self.api_type != "ollama":
            ctx = self._guess_context_from_model_name()
            self._detected_context_length = ctx
            return ctx

        try:
            payload = {"name": self.model}
            data = self._post(f"{self.base_url}/api/show", payload)

            # Method 1: model_info — look for context_length key
            model_info = data.get("model_info", {})
            for key in model_info:
                if "context_length" in key.lower():
                    ctx = model_info[key]
                    if isinstance(ctx, (int, float)) and ctx > 0:
                        self._detected_context_length = int(ctx)
                        return self._detected_context_length

            # Method 2: parameters string (num_ctx)
            params_str = data.get("parameters", "")
            if params_str:
                for line in params_str.split("\n"):
                    if "num_ctx" in line.lower():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                self._detected_context_length = int(parts[-1])
                                return self._detected_context_length
                            except ValueError:
                                pass

        except Exception:
            pass

        # Fallback
        ctx = self._guess_context_from_model_name()
        self._detected_context_length = ctx
        return ctx

    def _guess_context_from_model_name(self) -> int:
        """Guess context size from model name when API detection fails."""
        name = self.model.lower().split(":")[0]
        if name in MODEL_CONTEXT_DEFAULTS:
            return MODEL_CONTEXT_DEFAULTS[name]
        for prefix, ctx in MODEL_CONTEXT_DEFAULTS.items():
            if name.startswith(prefix):
                return ctx
        return 8192  # Safe minimum

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~3.5 chars per token for mixed PL/EN."""
        return max(1, len(text) // 4)

    def get_available_context(self) -> int:
        """How many tokens are available for new content (excluding response budget)."""
        used = self.estimate_tokens(self.system_prompt)
        for msg in self.conversation_history:
            used += self.estimate_tokens(msg.get("content", ""))
        available = self.context_length - used - self.max_tokens
        return max(512, available)

    def get_context_budget(self) -> dict:
        """Return a detailed breakdown of context budget allocation.
        Used by agent/server to decide how much space for memory, skills, etc."""
        total = self.context_length
        response = self.max_tokens
        system_used = self.estimate_tokens(self.system_prompt)
        history_used = sum(
            self.estimate_tokens(m.get("content", ""))
            for m in self.conversation_history
        )
        used = system_used + history_used + response
        free = max(0, total - used)

        return {
            "total": total,
            "response": response,
            "system_prompt": system_used,
            "history": history_used,
            "free": free,
            # Recommended allocations — fill the context aggressively
            "memory_budget":  max(500, free // 3),
            "skills_budget":  max(300, free // 5),
            "user_msg_budget": max(500, free // 3),
            "history_budget": max(200, total // 3),
        }

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def chat(self, user_message: str, include_history: bool = True) -> str:
        """Send a message and get a response (non-streaming)."""
        messages = self._build_messages(user_message, include_history)
        if self.api_type == "openai":
            return self._chat_openai(messages)
        else:
            return self._chat_ollama(messages)

    def chat_stream(self, user_message: str,
                    include_history: bool = True) -> Generator[str, None, None]:
        """Send a message and stream the response token by token."""
        messages = self._build_messages(user_message, include_history)
        if self.api_type == "openai":
            yield from self._stream_openai(messages)
        else:
            yield from self._stream_ollama(messages)

    def _build_messages(self, user_message: str,
                        include_history: bool) -> list[dict]:
        """Build messages array, filling context window as much as possible."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Use 95% of context — leave only small margin for overhead
        budget = int(self.context_length * 0.95) - self.max_tokens
        used = self.estimate_tokens(self.system_prompt) if self.system_prompt else 0
        user_tokens = self.estimate_tokens(user_message)
        used += user_tokens

        if include_history and self.conversation_history:
            # Fill with as much history as fits (up to 100 messages)
            history_to_add = []
            for msg in reversed(self.conversation_history[-100:]):
                msg_tokens = self.estimate_tokens(msg.get("content", ""))
                if used + msg_tokens < budget:
                    history_to_add.insert(0, msg)
                    used += msg_tokens
                else:
                    break
            messages.extend(history_to_add)

        messages.append({"role": "user", "content": user_message})
        return messages

    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        # Keep generous history — context budget will trim at send time
        if len(self.conversation_history) > 200:
            self.conversation_history = self.conversation_history[-150:]

    def clear_history(self):
        self.conversation_history = []

    # —— Ollama Native API (/api/chat) ——

    def _chat_ollama(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.context_length,
            },
        }
        try:
            data = self._post(f"{self.base_url}/api/chat", payload)
            content = data.get("message", {}).get("content", "")
            self.add_to_history("user", messages[-1]["content"])
            self.add_to_history("assistant", content)
            return content
        except urllib.error.HTTPError as e:
            return self._handle_http_error(e, "chat")
        except Exception as e:
            return f"[LLM Error] {e}"

    def _stream_ollama(self, messages: list[dict]) -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": self.context_length,
            },
        }
        full_response = ""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    if line:
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                full_response += token
                                yield token
                        except json.JSONDecodeError:
                            continue

            self.add_to_history("user", messages[-1]["content"])
            self.add_to_history("assistant", full_response)
        except urllib.error.HTTPError as e:
            yield self._handle_http_error(e, "stream")
        except Exception as e:
            yield f"\n[LLM Error] {e}"

    # —— OpenAI-compatible API (/v1/chat/completions) ——

    def _chat_openai(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        try:
            data = self._post(
                f"{self.base_url}/v1/chat/completions", payload
            )
            content = data["choices"][0]["message"]["content"]
            self.add_to_history("user", messages[-1]["content"])
            self.add_to_history("assistant", content)
            return content
        except urllib.error.HTTPError as e:
            return self._handle_http_error(e, "chat")
        except Exception as e:
            return f"[LLM Error] {e}"

    def _stream_openai(self, messages: list[dict]) -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        full_response = ""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    decoded = line.decode("utf-8").strip()
                    if decoded.startswith("data: ") and decoded != "data: [DONE]":
                        try:
                            chunk = json.loads(decoded[6:])
                            delta = chunk["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                full_response += token
                                yield token
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

            self.add_to_history("user", messages[-1]["content"])
            self.add_to_history("assistant", full_response)
        except urllib.error.HTTPError as e:
            yield self._handle_http_error(e, "stream")
        except Exception as e:
            yield f"\n[LLM Error] {e}"

    # —— Error handling ——

    def _handle_http_error(self, error: urllib.error.HTTPError, context: str) -> str:
        body = ""
        try:
            body = error.read().decode("utf-8", errors="replace")
        except Exception:
            pass

        if error.code == 404:
            return (
                f"\n[Error 404] Model \"{self.model}\" not found in Ollama.\n"
                f"  Model not pulled. Use command:\n"
                f"    /model {self.model}\n"
                f"  (will auto-pull if not available)"
            )
        elif error.code == 400:
            return f"\n[Error 400] Bad request: {body[:200]}"
        elif error.code == 500:
            return f"\n[Error 500] Ollama server error: {body[:200]}"
        else:
            return f"\n[HTTP {error.code}] {body[:200]}"

    # —— Utility ——

    def _post(self, url: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def list_models(self) -> list[str]:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def pull_model(self, model_name: str = None) -> Generator[str, None, None]:
        model = model_name or self.model
        payload = {"name": model, "stream": True}
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/pull",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                for line in resp:
                    if line:
                        try:
                            chunk = json.loads(line.decode("utf-8"))
                            st = chunk.get("status", "")
                            total = chunk.get("total", 0)
                            completed = chunk.get("completed", 0)
                            if total > 0:
                                pct = int(completed / total * 100)
                                yield f"\r  📥 {st}: {pct}%"
                            else:
                                yield f"\r  📥 {st}"
                        except json.JSONDecodeError:
                            continue
            yield f"\n  ✅ Model {model} ready!\n"
        except Exception as e:
            yield f"\n  ❌ Pull error: {e}\n"

    def is_model_available(self, model_name: str = None) -> bool:
        model = model_name or self.model
        models = self.list_models()
        return model in models or any(model in m for m in models)

    def check_connection(self) -> dict:
        """Check if Ollama is reachable, auto-detect context length."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                model_available = (
                    self.model in models
                    or any(self.model in m for m in models)
                )

                # Auto-detect context length on successful connection
                if model_available:
                    self.detect_context_length()

                return {
                    "connected": True,
                    "url": self.base_url,
                    "models": models,
                    "current_model": self.model,
                    "model_available": model_available,
                    "context_length": self.context_length,
                }
        except urllib.error.URLError as e:
            return {"connected": False, "url": self.base_url, "error": str(e)}
        except Exception as e:
            return {"connected": False, "url": self.base_url, "error": str(e)}