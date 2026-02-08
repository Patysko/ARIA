################################################################################
# ARIA Agent — Dockerfile
#
# Uses EXTERNAL Ollama instance (no Ollama inside container).
#
# Build & run:
#   docker compose up -d
#
# Or manually:
#   docker build -t aria-agent .
#   docker run -it -p 8080:8080 aria-agent --web
#
################################################################################

FROM ubuntu:24.04

LABEL maintainer="ARIA Agent"
LABEL description="Autonomous Reflective Intelligence Agent with external Ollama LLM"

# ── Avoid interactive prompts ──
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ──
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        curl \
        ca-certificates \
        procps \
        jq \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user ──
RUN useradd -m -s /bin/bash aria
ENV HOME=/home/aria

# ── Copy agent code ──
WORKDIR /home/aria/aria-agent
COPY --chown=aria:aria . .

# ── Ensure Python package init exists ──
RUN touch /home/aria/aria-agent/core/__init__.py \
    && touch /home/aria/aria-agent/web/__init__.py

EXPOSE 8080

# ── Create persistent directories ──
RUN mkdir -p /home/aria/aria-agent/memory \
             /home/aria/aria-agent/skills \
             /home/aria/aria-workspace \
    && chown -R aria:aria /home/aria

# ── Entrypoint ──
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]