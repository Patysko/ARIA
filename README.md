# ARIA — Autonomous Reflective Intelligence Agent

AI agent with self-improvement, compressed memory, and an OpenClaw-style skill system. Runs locally with Ollama. Vibecoded with Claude Opus 4.6 and tested with codegemma:latest.

## Architecture

```
aria-agent/
├── main.py                  # Entry point
├── config.json              # Configuration
├── core/
│   ├── agent.py             # Main orchestrator (Thread 1 + 2)
│   ├── config.py            # Configuration loader
│   ├── prompts.py           # All prompts & strings (PL/EN)
│   ├── memory.py            # Compressed memory (ST → LT)
│   ├── skills_manager.py    # Skill manager (SKILL.md format)
│   ├── computer.py          # System tools (shell, files, Python)
│   ├── llm.py               # Ollama/OpenAI LLM client
│   └── reflection.py        # Background reflection thread
├── web/
│   ├── server.py            # WebUI HTTP server + API
│   └── static/index.html    # React SPA dashboard
├── skills/                  # Skills (OpenClaw/AgentSkills format)
└── memory/                  # Persistent memory (auto-generated)
    ├── memory.json
    ├── reflections.jsonl
    └── self_model.json
```

## Two Thinking Threads

### Thread 1: Human Communication
- Interactive REPL (CLI) or WebUI with real-time dashboard
- Understands slash commands (`/exec`, `/python`, `/read`, `/write`, etc.)
- Chain of Thought: Analyze → Recall → Run skills → Plan → Answer
- Categorizes and weighs importance of every interaction

### Thread 2: Reflection & Self-Improvement
- Runs continuously in the background, independently from conversations
- Cycles through phases: introspection, pattern analysis, skill planning, skill building, skill testing, self-improvement, knowledge synthesis, exploration
- Autonomously creates, tests, and fixes skills
- Sends proactive messages when new skills are created
- Updates the agent's self-model

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

# View logs
docker compose logs -f aria
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

- Real-time chat with Chain of Thought (expandable)
- Thread 2 live thought stream panel
- Status dashboard with memory/skills/interaction stats
- Command palette with autocomplete (type `/`)
- Skill browser with run buttons
- Memory inspector
- Reflection timeline
- SSE-based real-time updates
- Proactive messages from Thread 2

## Inspiration

Architecture inspired by [OpenClaw](https://openclaw.ai/):
- SKILL.md-based skill system (AgentSkills format)
- Local command and script execution
- Persistent memory across sessions
- Modular, extensible architecture