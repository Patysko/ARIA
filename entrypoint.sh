#!/bin/bash
set -e

# ── Config ──
OLLAMA_HOST="${OLLAMA_HOST:-http://host.docker.internal:11434}"
MODEL="${ARIA_MODEL:-codegemma:latest}"
LANG="${ARIA_LANG:-pl}"

echo ""
echo "  ======================================="
echo "   ARIA Agent — Container Setup"
echo "  ======================================="
echo ""
echo "  Mode: EXTERNAL OLLAMA"
echo "  Ollama URL: $OLLAMA_HOST"
echo "  Model: $MODEL"
echo "  Language: $LANG"

# Update config.json with external URL, model, and language
python3 -c "
import json
with open('config.json') as f: cfg = json.load(f)
cfg['llm']['base_url'] = '$OLLAMA_HOST'
cfg['reflection_llm']['base_url'] = '$OLLAMA_HOST'
cfg['llm']['model'] = '$MODEL'
cfg['reflection_llm']['model'] = '$MODEL'
cfg.setdefault('agent', {})['language'] = '$LANG'
with open('config.json', 'w') as f: json.dump(cfg, f, indent=4)
print('  config.json updated')
"

# Wait for external Ollama
echo "  Waiting for Ollama..."
for i in $(seq 1 15); do
    if curl -sf "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
        echo "  OK Ollama available!"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "  WARNING: Cannot connect to $OLLAMA_HOST"
        echo "  Agent will start in offline mode"
    fi
    sleep 2
done

echo ""

# ── Launch ARIA ──
exec python3 main.py "$@"