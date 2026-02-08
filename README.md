# ARIA — Autonomous Reflective Intelligence Agent

AI agent with self-improvement, compressed memory, and an OpenClaw-style skill system. Runs locally with Ollama.

## Architecture

```
aria-agent/
├── main.py                  # Entry point
├── config.json              # Configuration
├── core/
│   ├── agent.py             # Main orchestrator (Thread 1 + 2)
│   ├── config.py            # Configuration loader
│   ├── prompts.py           # All prompts & strings (PL/EN)
│   ├── memory.py            # Compressed memory (ST → LT) + pending tasks
│   ├── skills_manager.py    # Skill manager (SKILL.md format) + error logs
│   ├── computer.py          # System tools (shell, files, Python)
│   ├── llm.py               # Ollama/OpenAI LLM client
│   ├── reflection.py        # Background reflection thread + task processing
│   └── logger.py            # Central event logging
├── web/
│   ├── server.py            # WebUI HTTP server + API
│   └── static/index.html    # React SPA dashboard
├── skills/                  # Skills (OpenClaw/AgentSkills format)
└── memory/                  # Persistent memory (auto-generated)
    ├── memory.json
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
- Categorizes and weighs importance of every interaction
- Detects when it can't answer and adds task to pending queue

### Thread 2: Reflection & Self-Improvement
- Runs continuously in the background, independently from conversations
- Cycles through phases: introspection, pattern analysis, skill planning, skill building, skill testing, self-improvement, knowledge synthesis, exploration, **pending tasks**
- Autonomously creates, tests, and fixes skills
- **Proactive messaging**: sends messages to user on any topic at any time (skill creation, task completion, observations, greetings)
- **Pending task processing**: retries user tasks that failed, notifies user when completed
- **Skill error log clearing**: clears error history after successful fix cycles
- Updates the agent's self-model

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

## Logging

All events are logged to `memory/logs/aria.log` (rotating, 10MB max, 3 backups):

```
2025-01-15 14:23:01 | INFO  | USER [5] | jak wygląda mój system?
2025-01-15 14:23:01 | DEBUG | COT/analyze | {"intent": "system info", ...}
2025-01-15 14:23:02 | INFO  | SKILL system-monitor/main.py exit=0 | CPU: 45%, RAM: 3.2GB...
2025-01-15 14:23:03 | INFO  | AGENT [5] skills=['system-monitor'] | Twój system wygląda...
2025-01-15 14:23:30 | INFO  | T2/skill_building | Cycle 12
2025-01-15 14:23:45 | INFO  | T2->USER | Stworzyłem nową umiejętność: disk-usage...
2025-01-15 14:24:00 | INFO  | TASK/completed task-1705312981-0 | Free space: 45GB...
```

Log categories: `USER`, `AGENT`, `COT/*`, `T2/*`, `T2->USER`, `SKILL`, `SKILL_ERR`, `TASK/*`, `CMD`, `ERROR`, `WARN`

## Compressed Memory

```
New interaction → Short-term memory (full detail)
                        ↓ (when buffer is full)
                  Thematic compression
                        ↓
                  Long-term memory (compressed blocks)
                        ↓
                  Episodic memory (key moments)
```

- **Short-term**: Full details, max 20 entries (configurable)
- **Long-term**: Compressed thematic blocks with keywords
- **Episodic**: Breakthrough moments (new skills, discoveries)
- **Pending tasks**: User requests awaiting completion
- Semantic search: `/recall <query>`

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

### Creating Skills
1. **Manually**: `/create-skill {json}` with name, description, instructions, script_code
2. **Automatically**: Thread 2 creates skills when it detects patterns
3. **Programmatically**: `skills_manager.create_skill(name, desc, instructions, scripts)`

## Language Support

ARIA supports **Polish (pl)** and **English (en)**. All prompts, LLM instructions, UI strings, and command descriptions switch based on language setting.

Set language via:
- **Environment variable**: `ARIA_LANG=en` (takes priority)
- **config.json**: `"agent": { "language": "en" }`
- **Docker Compose**: `ARIA_LANG=en` in environment section

## Quick Start (Docker)

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
        "max_tokens": 2048,
        "api_type": "ollama"
    },
    "reflection_llm": {
        "base_url": "http://localhost:11434",
        "model": "llama3.2",
        "temperature": 0.9,
        "max_tokens": 1024
    },
    "agent": {
        "language": "en",
        "reflection_interval": 3,
        "stream_responses": true,
        "short_term_limit": 20
    }
}
```

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
- Proactive messages from Thread 2 (skill creation, task completion, observations)
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