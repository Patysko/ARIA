# ARIA — Autonomous Reflective Intelligence Agent

AI agent with self-improvement, compressed memory, and an OpenClaw-style skill system. Runs locally with Ollama. Vibecoded with Claude Opus 4.6 and tested with codegemma:latest

## Architecture

```
aria-agent/
├── main.py                  # Entry point
├── config.json              # Configuration
├── core/
│   ├── agent.py             # Main orchestrator (Thread 1 + 2)
│   ├── config.py            # Configuration loader
│   ├── prompts.py           # All prompts & strings (PL/EN)
│   ├── memory.py            # Compressed memory (ST → LT) + RL neural network + pending tasks
│   ├── skills_manager.py    # Skill manager (SKILL.md format) + error logs
│   ├── computer.py          # System tools (shell, files, Python)
│   ├── llm.py               # Ollama/OpenAI LLM client (no timeouts, chunked responses)
│   ├── reflection.py        # Background reflection thread + system exploration + task processing
│   └── logger.py            # Central event logging
├── web/
│   ├── server.py            # WebUI HTTP server + API
│   └── static/index.html    # React SPA dashboard
├── skills/                  # Skills (OpenClaw/AgentSkills format)
└── memory/                  # Persistent memory (auto-generated)
    ├── memory.json
    ├── memory_network.json  # RL neural network weights
    ├── pending_tasks.json   # User tasks waiting for completion
    ├── reflections.jsonl
    ├── self_model.json
    └── logs/
        └── aria.log         # Full event log (rotating, 10MB)
```

## Two Thinking Threads

### Thread 1: Human Communication
- Interactive REPL (CLI) or WebUI with real-time dashboard
- Understands slash commands (`/exec`, `/python`, `/read`, `/write`, etc.)
- JSON-structured Chain of Thought: Analyze → Memory → Skill Selection → Execute → Plan → Answer
- Categorizes and weighs importance of every interaction (RL neural network scoring)
- Detects when it can't answer and adds task to pending queue
- Long responses auto-chunked across multiple messages (no truncation)

### Thread 2: Reflection & Self-Improvement
- Runs continuously in the background, independently from conversations
- Cycles through phases: introspection, pattern analysis, skill planning, skill building, skill testing, self-improvement, knowledge synthesis, exploration, pending tasks, **system exploration**
- Autonomously creates, **improves**, tests, and fixes skills (prefers improving over duplicating)
- **System exploration**: safely discovers system environment via read-only shell commands (double safety check: hardcoded blocklist + LLM verification)
- **Proactive messaging**: sends messages to user on any topic at any time (skill creation, task completion, observations, greetings)
- **Pending task processing**: retries user tasks that failed, notifies user when completed
- **Skill error log clearing**: clears error history after successful fix cycles
- Updates the agent's self-model

## LLM Communication (No Timeouts)

ARIA's LLM client has **zero timeouts** for Ollama — it relies entirely on Ollama's structured `done` flag to know when a response is complete. This eliminates mid-word truncation issues on slow hardware.

- **Streaming**: reads chunks until `{"done": true}` — no socket/connection timeout
- **Non-streaming**: internally uses streaming to avoid blocking
- **Truncation detection**: checks `done_reason == "length"` and auto-continues
- **`chat_long()`**: automatically splits responses truncated by token limit into multiple parts (up to 3), sending "Continue" to the model between parts

## Skill System: Improve Before Create

Thread 2's skill building phase now **prefers improving existing skills** over creating new duplicates:

1. LLM receives full code of all existing skills as context
2. LLM must choose `action: "improve"` or `action: "create"` with justification
3. **Duplicate detection**: Jaccard similarity check on name+description — rejects >50% overlap with existing skills
4. **Improvement workflow**: backup old code → write new → test → rollback on failure
5. **Creation workflow**: only when truly new functionality is needed

## System Exploration

Thread 2 periodically explores the host system to understand its environment:

1. LLM plans 3-5 read-only shell commands based on what it already knows
2. **Double safety check** before each command:
   - Hardcoded blocklist: `rm`, `sudo`, `dd`, `mkfs`, `shutdown`, `kill`, `chmod -R`, `> /`, etc.
   - LLM safety evaluation: assesses whether the command modifies files, installs packages, or could harm the system
3. Safe commands execute with 15s timeout
4. Results stored in memory as `system_discovery` category
5. Knowledge accumulates across cycles — LLM sees previous discoveries to avoid redundant exploration

## Pending Tasks System

When the agent detects it can't fulfill a user request (e.g., missing a required skill), it:

1. Adds the task to `pending_tasks.json` with the reason and missing capability
2. Notifies the user that the task has been queued
3. Thread 2 periodically checks pending tasks during the `pending_tasks` phase
4. When the capability is available (skill created/fixed), Thread 2 executes the task
5. User receives a proactive message with the result
6. Tasks are automatically cleaned up after 48 hours

## Skill Error Tracking

Each skill maintains an error log (in `.stats.json`):
- Errors are recorded automatically when scripts fail
- Thread 2 reads error logs to prioritize fixes
- Error logs are **cleared after successful fix cycles**
- View errors via `/skill <name>` or the Skills panel in WebUI

## Chain of Thought (JSON-Structured)

All internal reasoning steps use strict JSON format to prevent CoT artifacts from leaking into user-facing responses:

```
Step 1: Analyze → {"intent": "...", "skills": [...], "can_answer": true/false, ...}
Step 2: Memory recall → relevant context
Step 3: Skill selection → {"selections": [{"name": "...", "args": [...], "reason": "..."}]}
Step 4: Execute skills → raw output
Step 5: Plan → {"key_info": "...", "format": "...", "approach": "..."}
Step 6: Interpret → {"summary": "...", "useful": true/false}
Step 7: Final answer → natural language response (no CoT artifacts)
```

## RL Memory Network

Memory importance is scored by a lightweight reinforcement learning neural network that runs entirely on CPU:

```
Architecture: 28 features → 16 hidden (ReLU) → 1 output (sigmoid)
Training: REINFORCE-style policy gradient, single-step updates
Persistence: weights saved to memory/memory_network.json
```

**Features extracted** (28 total):
- Text metrics: length, word count, lexical diversity, question/exclamation density
- Content signals: has code, has URL, digit ratio, multiline, structured content
- Category one-hot encoding (8 categories)
- Metadata: interaction number, skill mentions, error mentions, source type

**RL Rewards**:
- Entry recalled by user query: **+1.0**
- User explicitly says "remember/zapamiętaj": **+2.0**
- Entry with high access count during compression: **+0.5**
- Entry never accessed with low importance: **-0.5**

**Scoring blend**: `final_importance = 0.4 × heuristic + 0.6 × neural_network`

The network improves over time — frequently useful memories get higher scores, low-value content gets filtered out.

## Compressed Memory (LLM-Powered)

```
New interaction → RL importance scoring → Short-term memory
                                               ↓ (buffer full)
                                    LLM-generated summaries
                                               ↓
                                    Long-term compressed blocks
                                               ↓
                                    Episodic memory (key moments)
```

- **Short-term**: Full details, max 20 entries, scored by RL network
- **Long-term**: LLM-generated summaries with max character limit (falls back to rule-based if LLM unavailable)
- **Episodic**: Breakthrough moments (new skills, discoveries)
- **Pending tasks**: User requests awaiting completion
- **RL training**: happens during compression (reward for accessed entries, penalty for never-used)
- Semantic search: `/recall <query>` (also triggers RL reward for recalled entries)

## Logging

All events are logged to `memory/logs/aria.log` (rotating, 10MB max, 3 backups):

```
2025-01-15 14:23:01 | INFO  | USER [5] | jak wygląda mój system?
2025-01-15 14:23:01 | DEBUG | COT/analyze | {"intent": "system info", ...}
2025-01-15 14:23:02 | INFO  | SKILL system-monitor/main.py exit=0 | CPU: 45%, RAM: 3.2GB...
2025-01-15 14:23:03 | INFO  | AGENT [5] skills=['system-monitor'] | Twój system wygląda...
2025-01-15 14:23:30 | INFO  | T2/system_exploration | Running: df -h (disk space)
2025-01-15 14:23:45 | INFO  | T2->USER | Ulepszyłem umiejętność: system-monitor...
2025-01-15 14:24:00 | INFO  | TASK/completed task-1705312981-0 | Free space: 45GB...
```

Log categories: `USER`, `AGENT`, `COT/*`, `T2/*`, `T2->USER`, `SKILL`, `SKILL_ERR`, `TASK/*`, `CMD`, `ERROR`, `WARN`

## Skill System (OpenClaw Format)

Each skill is a folder with `SKILL.md`:

```yaml
---
name: my-skill
description: When and why to use this skill
---

# Instructions in Markdown
...
```

### Creating & Improving Skills
1. **Manually**: `/create-skill {json}` with name, description, instructions, script_code
2. **Automatically**: Thread 2 creates or improves skills based on user interaction patterns
3. **Programmatically**: `skills_manager.create_skill(name, desc, instructions, scripts)`

## Language Support

ARIA supports **Polish (pl)** and **English (en)**. All prompts, LLM instructions, UI strings, and command descriptions switch based on language setting.

Set language via:
- **Environment variable**: `ARIA_LANG=en` (takes priority)
- **config.json**: `"agent": { "language": "en" }`
- **Docker Compose**: `ARIA_LANG=en` in environment section

## Quick Start (Docker) RECOMMENDED

```bash
# Prerequisites: Ollama running on host with a model pulled
ollama pull llama3.2

# Start ARIA (Polish, default)
docker compose up -d

# Start ARIA in English
ARIA_LANG=en docker compose up -d

# Start with a different model
ARIA_MODEL=qwen3:8b docker compose up -d

# Open WebUI
open http://localhost:8080

# View agent logs
docker compose logs -f aria
# Or check the detailed log file: memory/logs/aria.log
```

## Quick Start (Local)

```bash
# Install Ollama and pull a model
ollama pull llama3.2

# CLI mode
python main.py

# CLI with live Thread 2 output
python main.py --reflection

# WebUI on port 8080
python main.py --web

# WebUI on custom port
python main.py --web --port 3000
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ARIA_LANG` | `pl` | Language: `pl` or `en` |
| `ARIA_MODEL` | `codegemma:latest` | Ollama model name |
| `OLLAMA_HOST` | `http://host.docker.internal:11434` | Ollama server URL |

### config.json

```json
{
    "llm": {
        "base_url": "http://localhost:11434",
        "model": "llama3.2",
        "temperature": 0.7,
        "max_tokens": 4096,
        "context_length": "auto",
        "api_type": "ollama",
        "timeout": 300
    },
    "reflection_llm": {
        "base_url": "http://localhost:11434",
        "model": "llama3.2",
        "temperature": 0.9,
        "max_tokens": 4096,
        "context_length": "auto",
        "api_type": "ollama",
        "timeout": 180
    },
    "agent": {
        "language": "en",
        "reflection_interval": 3,
        "stream_responses": true,
        "short_term_limit": 20
    }
}
```

Note: `timeout` values in config are retained for non-Ollama endpoints (OpenAI-compatible). Ollama communication uses no timeouts — it relies on the structured `done` flag.

## Commands

| Command | Description |
|---------|-------------|
| `/help` | List all commands |
| `/status` | Agent status + Thread 2 info |
| `/memory` | Memory overview |
| `/recall <query>` | Search memory |
| `/skills` | List skills |
| `/skill <n>` | Skill details |
| `/run <skill>` | Run a skill script |
| `/create-skill <json>` | Create a new skill |
| `/tasks` | View pending tasks queue |
| `/thread2 [n]` | Show last n Thread 2 cycles |
| `/reflect` | Force a reflection cycle |
| `/exec <cmd>` | Execute shell command |
| `/python <code>` | Execute Python code |
| `/ls [path]` | List directory |
| `/read <path>` | Read file |
| `/write <path> <content>` | Write to file |
| `/sysinfo` | System information |
| `/compress` | Force memory compression |
| `/selfmodel` | Agent self-model |
| `/models` | List Ollama models |
| `/model <n>` | Switch active model |
| `/pull [model]` | Pull model from Ollama |
| `/ollama` | Ollama connection status |

## WebUI Features

- Real-time chat with JSON-structured Chain of Thought (expandable)
- Thread 2 live thought stream panel
- Proactive messages from Thread 2 (skill creation/improvement, task completion, system discoveries)
- Status dashboard with memory/skills/interaction/task stats
- Command palette with autocomplete (type `/`)
- Skill browser with run buttons and error counts
- Memory inspector
- Reflection timeline
- SSE-based real-time updates

## Inspiration

Architecture inspired by [OpenClaw](https://openclaw.ai/):
- SKILL.md-based skill system (AgentSkills format)
- Local command and script execution
- Persistent memory across sessions
- Modular, extensible architecture