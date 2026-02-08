"""
Compressed Memory System for ARIA.

Architecture:
- Short-term: Recent interactions with full detail
- Long-term: Compressed semantic blocks (themes, patterns, facts)
- Skills memory: What skills exist and how they performed
- Episodic: Key moments and breakthroughs

Memory filtering:
- Only meaningful user interactions go to memory
- Commands, model changes, system outputs are excluded
- Recall uses relevance scoring to select what goes into LLM context
"""

import json
import time
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Categories that should NOT be stored in agent memory
EXCLUDED_CATEGORIES = {"command_output", "system_internal"}

# Content patterns that indicate non-memorable content
NON_MEMORABLE_PATTERNS = [
    "/model ", "/models", "/pull ", "/ollama", "/status",
    "/help", "/sysinfo", "/compress", "/selfmodel",
]


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


class CompressedMemory:
    def __init__(self, config):
        self.config = config
        self.short_term: list[MemoryEntry] = []
        self.long_term: list[CompressedBlock] = []
        self.episodic: list[dict] = []  # Key moments
        self.compression_count = 0
        self._load()

    def _load(self):
        """Load memory from disk."""
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

    def save(self):
        """Persist memory to disk."""
        data = {
            "short_term": [e.to_dict() for e in self.short_term],
            "long_term": [b.to_dict() for b in self.long_term],
            "episodic": self.episodic,
            "compression_count": self.compression_count,
        }
        self.config.MEMORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @staticmethod
    def should_memorize(content: str, category: str = "general",
                        metadata: dict = None) -> bool:
        """Decide if this content is worth storing in memory.

        Filters out:
        - System commands and their outputs
        - Model/pull/status commands
        - Agent's own internal responses to commands
        - Very short meaningless inputs
        """
        if category in EXCLUDED_CATEGORIES:
            return False

        metadata = metadata or {}
        msg_type = metadata.get("type", "")

        # Don't store command outputs or system responses
        if msg_type in ("command_response", "system_internal"):
            return False

        content_lower = content.lower().strip()

        # Don't store slash commands that are system-management
        for pattern in NON_MEMORABLE_PATTERNS:
            if content_lower.startswith(pattern):
                return False

        # Don't store very short meaningless content
        if len(content_lower) < 3:
            return False

        return True

    def add(self, content: str, category: str = "general",
            importance: float = 0.5, metadata: dict = None):
        """Add a new memory to short-term buffer.

        Filters non-memorable content automatically.
        """
        if not self.should_memorize(content, category, metadata):
            return None

        entry = MemoryEntry(content, category, importance, metadata)
        self.short_term.append(entry)

        # Auto-compress when buffer is full
        if len(self.short_term) > self.config.SHORT_TERM_LIMIT:
            self.compress()

        self.save()
        return entry

    def add_episodic(self, event: str, significance: str):
        """Record a key moment/breakthrough."""
        self.episodic.append({
            "event": event,
            "significance": significance,
            "timestamp": time.time(),
        })
        if len(self.episodic) > 50:
            self.episodic = self.episodic[-50:]
        self.save()

    def compress(self) -> dict:
        """Compress older short-term memories into long-term blocks."""
        if len(self.short_term) <= 5:
            return {"compressed": 0, "blocks_created": 0}

        # Keep the most recent 5, compress the rest
        to_compress = self.short_term[:-5]
        self.short_term = self.short_term[-5:]

        # Group by category
        groups = defaultdict(list)
        for entry in to_compress:
            groups[entry.category].append(entry)

        blocks_created = 0
        for category, entries in groups.items():
            # Extract keywords (simple: most common significant words)
            all_words = " ".join(e.content for e in entries).lower().split()
            word_freq = defaultdict(int)
            for w in all_words:
                if len(w) > 3:
                    word_freq[w] += 1
            keywords = sorted(word_freq, key=word_freq.get, reverse=True)[:10]

            # Build compressed summary
            contents = [e.content[:80] for e in entries]
            summary = f"[{len(entries)} interakcji] " + " → ".join(contents)
            if len(summary) > 500:
                summary = summary[:497] + "..."

            max_imp = max(e.importance for e in entries)

            block = CompressedBlock(category, summary, keywords, len(entries), max_imp)
            self.long_term.append(block)
            blocks_created += 1

        # Trim long-term if too big
        if len(self.long_term) > self.config.MAX_LONG_TERM:
            self.long_term.sort(key=lambda b: b.max_importance, reverse=True)
            self.long_term = self.long_term[:self.config.MAX_LONG_TERM]

        self.compression_count += 1
        self.save()

        return {
            "compressed": len(to_compress),
            "blocks_created": blocks_created,
            "total_long_term": len(self.long_term),
        }

    def recall(self, query: str, limit: int = 5) -> dict:
        """Search memories by query string."""
        q = query.lower()
        tokens = set(q.split())

        # Score short-term
        st_scored = []
        for entry in self.short_term:
            score = 0
            content_lower = entry.content.lower()
            if q in content_lower:
                score += 3
            score += sum(1 for t in tokens if t in content_lower)
            score += entry.importance
            if score > 0:
                entry.access_count += 1
                st_scored.append((score, entry))

        # Score long-term
        lt_scored = []
        for block in self.long_term:
            score = 0
            summary_lower = block.summary.lower()
            if q in summary_lower:
                score += 2
            score += sum(1 for t in tokens if t in summary_lower)
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
        """Get memory context that fits within a token budget.

        Args:
            exclude_sources: list of metadata 'source' values to skip
                           (e.g. ["thread2"] to exclude Thread 2 reflections)
        """
        results = self.recall(query, limit=10)
        parts = []
        used_tokens = 0
        exclude_sources = set(exclude_sources or [])

        # High-importance short-term first
        for score, entry_dict in results.get("short_term", []):
            # Skip excluded sources
            meta = entry_dict.get("metadata", {})
            if meta.get("source") in exclude_sources:
                continue
            cat = entry_dict.get("category", "")
            if cat in ("self-reflection", "system_internal"):
                continue
            content = entry_dict.get("content", "")[:150]
            line = f"[{cat}] {content}"
            line_tokens = len(line) // 4
            if used_tokens + line_tokens > max_tokens:
                break
            parts.append(line)
            used_tokens += line_tokens

        # Then long-term
        for score, block_dict in results.get("long_term", []):
            cat = block_dict.get("category", "")
            if cat in ("self-reflection", "system_internal"):
                continue
            summary = block_dict.get("summary", "")[:150]
            line = f"[LT/{cat}] {summary}"
            line_tokens = len(line) // 4
            if used_tokens + line_tokens > max_tokens:
                break
            parts.append(line)
            used_tokens += line_tokens

        # Add recent non-scored entries if space remains
        if used_tokens < max_tokens * 0.7:
            for entry in reversed(self.short_term[-5:]):
                if entry.metadata.get("source") in exclude_sources:
                    continue
                if entry.category in ("self-reflection", "system_internal"):
                    continue
                if entry.content not in [p.split("] ", 1)[-1] if "] " in p else p for p in parts]:
                    line = f"[recent/{entry.category}] {entry.content[:100]}"
                    line_tokens = len(line) // 4
                    if used_tokens + line_tokens > max_tokens:
                        break
                    parts.append(line)
                    used_tokens += line_tokens

        return "\n".join(parts) if parts else "(brak powiazanych wspomnien)"

    def get_stats(self) -> dict:
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "episodic_count": len(self.episodic),
            "compression_count": self.compression_count,
            "categories": list(set(
                [e.category for e in self.short_term] +
                [b.category for b in self.long_term]
            )),
        }

    def get_context_summary(self, max_tokens: int = 800) -> str:
        """Get a compact memory summary for the agent's system prompt.

        Budget-aware: stays within max_tokens.
        """
        stats = self.get_stats()
        lines = [
            f"Pamięć: {stats['short_term_count']} krótkoterminowych, "
            f"{stats['long_term_count']} skompresowanych, "
            f"{stats['episodic_count']} epizodycznych",
        ]
        used = len(lines[0]) // 4

        if stats["categories"]:
            cat_line = f"Kategorie: {', '.join(stats['categories'][:8])}"
            used += len(cat_line) // 4
            lines.append(cat_line)

        # Include recent short-term, sorted by importance
        if self.short_term:
            recent = sorted(self.short_term[-10:],
                            key=lambda e: e.importance, reverse=True)
            lines.append("--- Ostatnie ważne wspomnienia ---")
            for entry in recent[:5]:
                content = entry.content[:100]
                line = f"  [{entry.category}] {content}"
                line_tokens = len(line) // 4
                if used + line_tokens > max_tokens:
                    break
                lines.append(line)
                used += line_tokens

        return "\n".join(lines)

    def reset(self):
        """Clear all memory."""
        self.short_term = []
        self.long_term = []
        self.episodic = []
        self.compression_count = 0
        self.save()