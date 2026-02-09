"""
Compressed Memory System for ARIA.

Architecture:
- Short-term: Recent interactions with full detail
- Long-term: Compressed semantic blocks (LLM-generated summaries)
- Episodic: Key moments and breakthroughs
- MemoryNetwork: Lightweight RL neural net for importance scoring

Memory filtering:
- MemoryNetwork scores importance of each entry (trained via RL)
- LLM generates compressed summaries during compression
- Only meaningful user interactions go to memory
"""

import json
import math
import random
import time
from pathlib import Path
from collections import defaultdict
from typing import Optional


# Categories that should NOT be stored in agent memory
EXCLUDED_CATEGORIES = {"command_output", "system_internal"}

NON_MEMORABLE_PATTERNS = [
    "/model ", "/models", "/pull ", "/ollama", "/status",
    "/help", "/sysinfo", "/compress", "/selfmodel",
]


# ============================================================
#  LIGHTWEIGHT RL NEURAL NETWORK FOR MEMORY IMPORTANCE
# ============================================================

class MemoryNetwork:
    """Tiny neural net (1 hidden layer) trained via reinforcement learning.
    Predicts importance score [0,1] for memory entries.
    Fast on CPU: ~50 features, 1 hidden layer of 16 neurons.

    Features extracted from text:
    - length, word count, unique words, question marks, exclamation
    - category one-hot (8 cats), has_code, has_url, has_numbers
    - time_of_day, is_user_input, keyword_hits (important words)

    RL signal: entries recalled often get reward +1, entries never accessed get -0.5,
    entries the user explicitly asked to remember get +2.
    """

    CATEGORIES = ["general", "memory", "skill", "code", "file", "system",
                  "agent_response", "self-reflection"]
    IMPORTANT_KEYWORDS = {
        "pamietaj", "wazne", "remember", "important", "critical", "always",
        "never", "nigdy", "zawsze", "preferuj", "prefer", "nauczyc", "learn",
        "zanotuj", "note", "klucz", "key", "priority", "priorytet",
    }
    N_FEATURES = 28  # Fixed feature count
    N_HIDDEN = 16

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        # Xavier-ish init
        scale1 = math.sqrt(2.0 / self.N_FEATURES)
        scale2 = math.sqrt(2.0 / self.N_HIDDEN)
        self.w1 = [[random.gauss(0, scale1) for _ in range(self.N_FEATURES)]
                    for _ in range(self.N_HIDDEN)]
        self.b1 = [0.0] * self.N_HIDDEN
        self.w2 = [random.gauss(0, scale2) for _ in range(self.N_HIDDEN)]
        self.b2 = 0.0
        self.lr = 0.01
        self.train_count = 0
        self._load()

    def _load(self):
        if self.model_path and self.model_path.exists():
            try:
                d = json.loads(self.model_path.read_text())
                self.w1 = d["w1"]
                self.b1 = d["b1"]
                self.w2 = d["w2"]
                self.b2 = d["b2"]
                self.lr = d.get("lr", 0.01)
                self.train_count = d.get("train_count", 0)
            except Exception:
                pass

    def save(self):
        if self.model_path:
            try:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model_path.write_text(json.dumps({
                    "w1": self.w1, "b1": self.b1,
                    "w2": self.w2, "b2": self.b2,
                    "lr": self.lr, "train_count": self.train_count,
                }, ensure_ascii=False))
            except Exception:
                pass

    def extract_features(self, content: str, category: str = "general",
                         metadata: dict = None) -> list[float]:
        """Extract fixed-size feature vector from memory entry."""
        metadata = metadata or {}
        t = content.lower()
        words = t.split()
        n_words = len(words)
        unique_words = len(set(words))

        features = [
            min(len(content) / 500.0, 1.0),           # 0: normalized length
            min(n_words / 100.0, 1.0),                 # 1: word count
            unique_words / max(n_words, 1),             # 2: lexical diversity
            t.count("?") / max(n_words, 1),             # 3: question density
            t.count("!") / max(n_words, 1),             # 4: exclamation density
            1.0 if "```" in content or "def " in t else 0.0,  # 5: has code
            1.0 if "http" in t or "www" in t else 0.0,  # 6: has URL
            sum(1 for c in content if c.isdigit()) / max(len(content), 1),  # 7: digit ratio
            1.0 if metadata.get("type") == "user_input" else 0.0,  # 8: is user input
            sum(1 for kw in self.IMPORTANT_KEYWORDS if kw in t) / 5.0,  # 9: keyword hits
            min(content.count("\n") / 10.0, 1.0),       # 10: multiline
            1.0 if any(c.isupper() for c in content[:20]) else 0.0,  # 11: starts with caps
        ]

        # 12-19: category one-hot
        for cat in self.CATEGORIES:
            features.append(1.0 if category == cat else 0.0)

        # 20-27: additional metadata signals
        features.append(min(metadata.get("n", 0) / 100.0, 1.0))  # interaction number
        features.append(1.0 if "skill" in t else 0.0)
        features.append(1.0 if "error" in t or "blad" in t else 0.0)
        features.append(1.0 if category == "skill-creation" else 0.0)
        features.append(1.0 if len(content) > 200 else 0.0)
        features.append(1.0 if metadata.get("source") == "thread2" else 0.0)
        features.append(min(t.count(",") / 5.0, 1.0))  # complexity proxy
        features.append(1.0 if ":" in content else 0.0)  # structured content

        # Ensure exact size
        features = features[:self.N_FEATURES]
        while len(features) < self.N_FEATURES:
            features.append(0.0)
        return features

    def predict(self, features: list[float]) -> float:
        """Forward pass → importance score [0, 1]."""
        hidden = []
        for j in range(self.N_HIDDEN):
            s = self.b1[j]
            for i in range(self.N_FEATURES):
                s += self.w1[j][i] * features[i]
            hidden.append(max(0.0, s))  # ReLU
        out = self.b2
        for j in range(self.N_HIDDEN):
            out += self.w2[j] * hidden[j]
        return 1.0 / (1.0 + math.exp(-max(-10, min(10, out))))  # Sigmoid

    def score(self, content: str, category: str = "general",
              metadata: dict = None) -> float:
        """Score importance of a memory entry."""
        features = self.extract_features(content, category, metadata)
        return self.predict(features)

    def train(self, content: str, category: str, metadata: dict,
              reward: float):
        """Single-step RL update (REINFORCE-style).
        reward > 0: entry was useful (recalled, accessed, user asked to remember)
        reward < 0: entry was not useful (never accessed, low quality)
        """
        features = self.extract_features(content, category, metadata)

        # Forward
        hidden = []
        for j in range(self.N_HIDDEN):
            s = self.b1[j]
            for i in range(self.N_FEATURES):
                s += self.w1[j][i] * features[i]
            hidden.append(max(0.0, s))  # ReLU
        out = self.b2
        for j in range(self.N_HIDDEN):
            out += self.w2[j] * hidden[j]
        pred = 1.0 / (1.0 + math.exp(-max(-10, min(10, out))))

        # Policy gradient: d_loss = reward * (target - pred)
        target = max(0.0, min(1.0, pred + reward * 0.3))
        error = target - pred
        grad_out = error * pred * (1 - pred)  # sigmoid derivative

        # Backprop to w2, b2
        lr = self.lr
        self.b2 += lr * grad_out
        for j in range(self.N_HIDDEN):
            self.w2[j] += lr * grad_out * hidden[j]

        # Backprop to w1, b1 (through ReLU)
        for j in range(self.N_HIDDEN):
            if hidden[j] > 0:  # ReLU gate
                grad_h = grad_out * self.w2[j]
                self.b1[j] += lr * grad_h
                for i in range(self.N_FEATURES):
                    self.w1[j][i] += lr * grad_h * features[i]

        self.train_count += 1
        if self.train_count % 50 == 0:
            self.save()


# ============================================================
#  MEMORY ENTRY & COMPRESSED BLOCK
# ============================================================

class MemoryEntry:
    def __init__(self, content: str, category: str = "general",
                 importance: float = 0.5, metadata: dict = None):
        self.content = content
        self.category = category
        self.importance = importance
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.access_count = 0

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "category": self.category,
            "importance": self.importance,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        entry = cls(d["content"], d.get("category", "general"),
                    d.get("importance", 0.5), d.get("metadata"))
        entry.timestamp = d.get("timestamp", time.time())
        entry.access_count = d.get("access_count", 0)
        return entry


class CompressedBlock:
    def __init__(self, category: str, summary: str, keywords: list,
                 source_count: int, max_importance: float):
        self.category = category
        self.summary = summary
        self.keywords = keywords
        self.source_count = source_count
        self.max_importance = max_importance
        self.created_at = time.time()

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "summary": self.summary,
            "keywords": self.keywords,
            "source_count": self.source_count,
            "max_importance": self.max_importance,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompressedBlock":
        block = cls(d["category"], d["summary"], d.get("keywords", []),
                    d.get("source_count", 1), d.get("max_importance", 0.5))
        block.created_at = d.get("created_at", time.time())
        return block


# ============================================================
#  COMPRESSED MEMORY
# ============================================================

class CompressedMemory:
    def __init__(self, config):
        self.config = config
        self.short_term: list[MemoryEntry] = []
        self.long_term: list[CompressedBlock] = []
        self.episodic: list[dict] = []
        self.compression_count = 0
        self.pending_tasks: list[dict] = []
        self._llm_client = None  # Set externally for LLM-based compression

        # RL memory network
        nn_path = config.MEMORY_DIR / "memory_network.json"
        self.network = MemoryNetwork(model_path=nn_path)

        self._load()
        self._load_tasks()

    def set_llm(self, llm_client):
        """Set LLM client for smart compression summaries."""
        self._llm_client = llm_client

    def _load(self):
        path = self.config.MEMORY_FILE
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self.short_term = [MemoryEntry.from_dict(e) for e in data.get("short_term", [])]
                self.long_term = [CompressedBlock.from_dict(b) for b in data.get("long_term", [])]
                self.episodic = data.get("episodic", [])
                self.compression_count = data.get("compression_count", 0)
            except (json.JSONDecodeError, KeyError):
                pass

    def _load_tasks(self):
        path = self.config.PENDING_TASKS_FILE
        if path.exists():
            try:
                self.pending_tasks = json.loads(path.read_text())
            except (json.JSONDecodeError, KeyError):
                self.pending_tasks = []

    def _save_tasks(self):
        self.config.PENDING_TASKS_FILE.write_text(
            json.dumps(self.pending_tasks, ensure_ascii=False, indent=2))

    def save(self):
        data = {
            "short_term": [e.to_dict() for e in self.short_term],
            "long_term": [b.to_dict() for b in self.long_term],
            "episodic": self.episodic,
            "compression_count": self.compression_count,
        }
        self.config.MEMORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # --- Filtering ---

    @staticmethod
    def should_memorize(content: str, category: str = "general",
                        metadata: dict = None) -> bool:
        if category in EXCLUDED_CATEGORIES:
            return False
        metadata = metadata or {}
        if metadata.get("type") in ("command_response", "system_internal"):
            return False
        c = content.strip()
        if len(c) < 3:
            return False
        if any(c.startswith(p) for p in NON_MEMORABLE_PATTERNS):
            return False
        return True

    # --- Add ---

    def add(self, content: str, category: str = "general",
            importance: float = 0.5, metadata: dict = None):
        """Add a new memory. Importance scored by RL network + heuristic blend."""
        if not self.should_memorize(content, category, metadata):
            return None

        # RL network score blended with heuristic
        nn_score = self.network.score(content, category, metadata)
        blended = 0.4 * importance + 0.6 * nn_score
        blended = max(0.05, min(1.0, blended))

        # Strong override: user explicitly says "remember"
        t = content.lower()
        if any(kw in t for kw in ("pamietaj", "zapamietaj", "remember", "wazne", "important")):
            blended = max(blended, 0.9)
            self.network.train(content, category, metadata or {}, reward=2.0)

        entry = MemoryEntry(content, category, blended, metadata)
        self.short_term.append(entry)

        if len(self.short_term) > self.config.SHORT_TERM_LIMIT:
            self.compress()
        self.save()
        return entry

    def add_episodic(self, event: str, significance: str):
        self.episodic.append({
            "event": event, "significance": significance,
            "timestamp": time.time(),
        })
        if len(self.episodic) > 50:
            self.episodic = self.episodic[-50:]
        self.save()

    # --- Compress with LLM summaries ---

    def compress(self) -> dict:
        """Compress older short-term memories into long-term blocks.
        Uses LLM for intelligent summaries when available."""
        if len(self.short_term) <= 5:
            return {"compressed": 0, "blocks_created": 0}

        to_compress = self.short_term[:-5]
        self.short_term = self.short_term[-5:]

        # Train RL network: entries with low access_count get negative reward
        for entry in to_compress:
            if entry.access_count > 0:
                self.network.train(entry.content, entry.category,
                                   entry.metadata, reward=0.5)
            elif entry.importance < 0.3:
                self.network.train(entry.content, entry.category,
                                   entry.metadata, reward=-0.5)

        # Group by category
        groups = defaultdict(list)
        for entry in to_compress:
            groups[entry.category].append(entry)

        blocks_created = 0
        for category, entries in groups.items():
            keywords = self._extract_keywords(entries)
            summary = self._make_summary(category, entries)
            max_imp = max(e.importance for e in entries)
            block = CompressedBlock(category, summary, keywords,
                                    len(entries), max_imp)
            self.long_term.append(block)
            blocks_created += 1

        if len(self.long_term) > self.config.MAX_LONG_TERM:
            self.long_term.sort(key=lambda b: b.max_importance, reverse=True)
            self.long_term = self.long_term[:self.config.MAX_LONG_TERM]

        self.compression_count += 1
        self.network.save()
        self.save()
        return {
            "compressed": len(to_compress),
            "blocks_created": blocks_created,
            "total_long_term": len(self.long_term),
        }

    def _extract_keywords(self, entries: list[MemoryEntry]) -> list[str]:
        all_words = " ".join(e.content for e in entries).lower().split()
        freq = defaultdict(int)
        for w in all_words:
            if len(w) > 3:
                freq[w] += 1
        return sorted(freq, key=freq.get, reverse=True)[:10]

    def _make_summary(self, category: str, entries: list[MemoryEntry],
                      max_len: int = 300) -> str:
        """Generate summary. Uses LLM if available, otherwise rule-based."""
        contents = [e.content[:120] for e in entries]
        raw = "\n".join(f"- {c}" for c in contents)

        if self._llm_client:
            try:
                from core.prompts import get_lang
                lang = get_lang()
                if lang == "pl":
                    prompt = (
                        f"Streszcz ponizsze {len(entries)} interakcji z kategorii '{category}' "
                        f"w maksymalnie {max_len} znakow. Zachowaj kluczowe fakty. "
                        f"Odpowiedz TYLKO streszczeniem, bez komentarzy.\n\n{raw}"
                    )
                else:
                    prompt = (
                        f"Summarize these {len(entries)} interactions from category '{category}' "
                        f"in at most {max_len} characters. Keep key facts. "
                        f"Respond ONLY with the summary, no commentary.\n\n{raw}"
                    )
                summary = self._llm_client.chat(prompt, include_history=False)
                summary = summary.strip()
                if len(summary) > max_len:
                    summary = summary[:max_len - 3] + "..."
                if summary and len(summary) > 10:
                    return summary
            except Exception:
                pass

        # Fallback: rule-based
        summary = f"[{len(entries)}x] " + " | ".join(c[:60] for c in contents)
        if len(summary) > max_len:
            summary = summary[:max_len - 3] + "..."
        return summary

    # --- Recall ---

    def recall(self, query: str, limit: int = 5) -> dict:
        q = query.lower()
        tokens = set(q.split())

        st_scored = []
        for entry in self.short_term:
            score = 0
            cl = entry.content.lower()
            if q in cl:
                score += 3
            score += sum(1 for t in tokens if t in cl)
            score += entry.importance
            if score > 0:
                entry.access_count += 1
                # RL reward for being recalled
                self.network.train(entry.content, entry.category,
                                   entry.metadata, reward=1.0)
                st_scored.append((score, entry))

        lt_scored = []
        for block in self.long_term:
            score = 0
            sl = block.summary.lower()
            if q in sl:
                score += 2
            score += sum(1 for t in tokens if t in sl)
            score += sum(2 for kw in block.keywords if kw in tokens)
            score += block.max_importance
            if score > 0:
                lt_scored.append((score, block))

        st_scored.sort(key=lambda x: x[0], reverse=True)
        lt_scored.sort(key=lambda x: x[0], reverse=True)
        return {
            "short_term": [(s, e.to_dict()) for s, e in st_scored[:limit]],
            "long_term": [(s, b.to_dict()) for s, b in lt_scored[:limit]],
        }

    def get_relevant_context(self, query: str, max_tokens: int = 1500,
                             exclude_sources: list = None) -> str:
        results = self.recall(query, limit=10)
        parts = []
        used_tokens = 0
        exclude_sources = set(exclude_sources or [])

        for score, entry_dict in results.get("short_term", []):
            meta = entry_dict.get("metadata", {})
            if meta.get("source") in exclude_sources:
                continue
            cat = entry_dict.get("category", "")
            if cat in ("self-reflection", "system_internal"):
                continue
            content = entry_dict.get("content", "")[:150]
            line = f"[{cat}] {content}"
            lt = len(line) // 4
            if used_tokens + lt > max_tokens:
                break
            parts.append(line)
            used_tokens += lt

        for score, block_dict in results.get("long_term", []):
            cat = block_dict.get("category", "")
            if cat in ("self-reflection", "system_internal"):
                continue
            summary = block_dict.get("summary", "")[:150]
            line = f"[LT/{cat}] {summary}"
            lt = len(line) // 4
            if used_tokens + lt > max_tokens:
                break
            parts.append(line)
            used_tokens += lt

        if used_tokens < max_tokens * 0.7:
            for entry in reversed(self.short_term[-5:]):
                if entry.metadata.get("source") in exclude_sources:
                    continue
                if entry.category in ("self-reflection", "system_internal"):
                    continue
                line = f"[recent/{entry.category}] {entry.content[:100]}"
                lt = len(line) // 4
                if used_tokens + lt > max_tokens:
                    break
                parts.append(line)
                used_tokens += lt

        if parts:
            return "\n".join(parts)
        from core.prompts import get_lang, MEM
        return MEM.get(get_lang(), MEM["en"])["no_relevant"]

    # --- Pending tasks ---

    def add_task(self, user_message: str, reason: str,
                 needed_skill: str = "", interaction_n: int = 0) -> dict:
        task = {
            "id": f"task-{int(time.time())}-{len(self.pending_tasks)}",
            "message": user_message, "reason": reason,
            "needed_skill": needed_skill,
            "created_at": time.time(), "attempts": 0,
            "status": "pending", "interaction_n": interaction_n,
        }
        self.pending_tasks.append(task)
        self._save_tasks()
        return task

    def get_pending_tasks(self) -> list[dict]:
        return [t for t in self.pending_tasks if t["status"] == "pending"]

    def mark_task(self, task_id: str, status: str, result: str = ""):
        for t in self.pending_tasks:
            if t["id"] == task_id:
                t["status"] = status
                t["completed_at"] = time.time()
                t["result"] = result[:500]
                t["attempts"] = t.get("attempts", 0) + 1
                break
        self._save_tasks()

    def increment_task_attempt(self, task_id: str):
        for t in self.pending_tasks:
            if t["id"] == task_id:
                t["attempts"] = t.get("attempts", 0) + 1
                break
        self._save_tasks()

    def cleanup_tasks(self, max_age_hours: float = 24):
        cutoff = time.time() - max_age_hours * 3600
        self.pending_tasks = [
            t for t in self.pending_tasks
            if t["status"] == "pending" or t.get("completed_at", 0) > cutoff
        ]
        self._save_tasks()

    # --- Stats ---

    def get_stats(self) -> dict:
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "episodic_count": len(self.episodic),
            "compression_count": self.compression_count,
            "pending_tasks_count": len(self.get_pending_tasks()),
            "nn_train_count": self.network.train_count,
            "categories": list(set(
                [e.category for e in self.short_term] +
                [b.category for b in self.long_term]
            )),
        }

    def get_context_summary(self, max_tokens: int = 800) -> str:
        from core.prompts import get_lang, MEM
        lang = get_lang()
        m = MEM.get(lang, MEM["en"])
        stats = self.get_stats()
        lines = [m["memory_stats"].format(
            st=stats['short_term_count'], lt=stats['long_term_count'],
            ep=stats['episodic_count'])]
        used = len(lines[0]) // 4
        if stats["categories"]:
            cat_line = m["categories_label"] + ', '.join(stats['categories'][:8])
            used += len(cat_line) // 4
            lines.append(cat_line)
        if self.short_term:
            recent = sorted(self.short_term[-10:],
                            key=lambda e: e.importance, reverse=True)
            lines.append(m["recent_label"])
            for entry in recent[:5]:
                line = f"  [{entry.category}] {entry.content[:100]}"
                lt = len(line) // 4
                if used + lt > max_tokens:
                    break
                lines.append(line)
                used += lt
        return "\n".join(lines)

    def reset(self):
        self.short_term = []
        self.long_term = []
        self.episodic = []
        self.compression_count = 0
        self.save()