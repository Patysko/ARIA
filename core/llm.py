"""
LLM Client for ARIA — connects to Ollama or OpenAI-compatible endpoints.

Features:
- Context length awareness (token budget management)
- Ollama native API + OpenAI-compatible mode
- Streaming support
- Model pull integrated into model switching
"""

import json
import sys
import urllib.request
import urllib.error
from typing import Generator


class LLMClient:
    """Connects ARIA to a local LLM via Ollama."""

    def __init__(self, config: dict):
        self.base_url = config.get("base_url", "http://host.docker.internal:11434")
        self.model = config.get("model", "codegemma:latest")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2048)
        self.context_length = config.get("context_length", 8192)
        self.api_type = config.get("api_type", "ollama")
        self.timeout = config.get("timeout", 120)
        self.conversation_history: list[dict] = []
        self.system_prompt = ""

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~3.5 chars per token for mixed PL/EN."""
        return max(1, len(text) // 4)

    def get_available_context(self) -> int:
        """How many tokens are available for new content."""
        used = self.estimate_tokens(self.system_prompt)
        for msg in self.conversation_history:
            used += self.estimate_tokens(msg.get("content", ""))
        # Reserve space for response
        available = self.context_length - used - self.max_tokens
        return max(512, available)

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
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Calculate token budget
        budget = self.context_length - self.max_tokens
        used = self.estimate_tokens(self.system_prompt) if self.system_prompt else 0
        user_tokens = self.estimate_tokens(user_message)
        used += user_tokens

        if include_history and self.conversation_history:
            # Add history from most recent, fitting within budget
            history_to_add = []
            for msg in reversed(self.conversation_history[-20:]):
                msg_tokens = self.estimate_tokens(msg.get("content", ""))
                if used + msg_tokens < budget * 0.85:  # Leave 15% margin
                    history_to_add.insert(0, msg)
                    used += msg_tokens
                else:
                    break
            messages.extend(history_to_add)

        messages.append({"role": "user", "content": user_message})
        return messages

    def add_to_history(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        if len(self.conversation_history) > 40:
            self.conversation_history = self.conversation_history[-30:]

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
            return f"[Błąd LLM] {e}"

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
            yield f"\n[Błąd LLM] {e}"

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
            return f"[Błąd LLM] {e}"

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
            yield f"\n[Błąd LLM] {e}"

    # —— Error handling ——

    def _handle_http_error(self, error: urllib.error.HTTPError, context: str) -> str:
        body = ""
        try:
            body = error.read().decode("utf-8", errors="replace")
        except Exception:
            pass

        if error.code == 404:
            return (
                f"\n[Błąd 404] Model \"{self.model}\" nie znaleziony w Ollama.\n"
                f"  Model nie jest pobrany. Użyj komendy:\n"
                f"    /model {self.model}\n"
                f"  (automatycznie pobierze model jeśli nie istnieje)"
            )
        elif error.code == 400:
            return f"\n[Błąd 400] Nieprawidłowe zapytanie: {body[:200]}"
        elif error.code == 500:
            return f"\n[Błąd 500] Błąd serwera Ollama: {body[:200]}"
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
        """List available models on the Ollama server."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def pull_model(self, model_name: str = None) -> Generator[str, None, None]:
        """Pull a model from Ollama registry, yielding progress."""
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
                            status = chunk.get("status", "")
                            total = chunk.get("total", 0)
                            completed = chunk.get("completed", 0)
                            if total > 0:
                                pct = int(completed / total * 100)
                                yield f"\r  📥 {status}: {pct}%"
                            else:
                                yield f"\r  📥 {status}"
                        except json.JSONDecodeError:
                            continue
            yield f"\n  ✅ Model {model} pobrany!\n"
        except Exception as e:
            yield f"\n  ❌ Błąd pobierania: {e}\n"

    def is_model_available(self, model_name: str = None) -> bool:
        """Check if a specific model is available locally."""
        model = model_name or self.model
        models = self.list_models()
        return model in models or any(model in m for m in models)

    def check_connection(self) -> dict:
        """Check if Ollama is reachable and model is available."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", [])]
                model_available = (
                    self.model in models
                    or any(self.model in m for m in models)
                )
                return {
                    "connected": True,
                    "url": self.base_url,
                    "models": models,
                    "current_model": self.model,
                    "model_available": model_available,
                }
        except urllib.error.URLError as e:
            return {"connected": False, "url": self.base_url, "error": str(e)}
        except Exception as e:
            return {"connected": False, "url": self.base_url, "error": str(e)}