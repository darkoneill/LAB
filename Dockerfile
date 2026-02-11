# ============================================================
# OpenClaw NexusMind - Dockerfile
# Multi-stage build optimise pour VPS (Hostinger / Ubuntu)
# ============================================================

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Dependances systeme pour la compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="OpenClaw Contributors"
LABEL description="OpenClaw NexusMind - Autonomous AI Assistant"
LABEL version="1.0.0"

# Arguments de build
ARG UID=1000
ARG GID=1000

# Dependances systeme runtime minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g ${GID} openclaw \
    && useradd -u ${UID} -g openclaw -m -s /bin/bash openclaw

# Copier les packages Python depuis le builder
COPY --from=builder /install /usr/local

# Creer les repertoires de l'application
RUN mkdir -p /app /data/memory /data/config /data/logs /data/traces /data/skills \
    && chown -R openclaw:openclaw /app /data

WORKDIR /app

# Copier le code source
COPY --chown=openclaw:openclaw openclaw/ ./openclaw/
COPY --chown=openclaw:openclaw run.py setup.py requirements.txt ./
COPY --chown=openclaw:openclaw docker/entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Volumes pour la persistance
VOLUME ["/data/memory", "/data/config", "/data/logs", "/data/traces", "/data/skills"]

# Ports : Gateway API + Web UI
EXPOSE 18789

# Variables d'environnement par defaut
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    OPENCLAW_DATA_DIR=/data \
    OPENCLAW_GATEWAY__HOST=0.0.0.0 \
    OPENCLAW_GATEWAY__PORT=18789 \
    OPENCLAW_LOGGING__LEVEL=INFO \
    OPENCLAW_LOGGING__FILE=/data/logs/openclaw.log \
    OPENCLAW_MEMORY__STORE_PATH=/data/memory \
    OPENCLAW_TRACING__STORE_PATH=/data/traces \
    OPENCLAW_SANDBOX__WORKSPACE_PATH=/workspace \
    OPENCLAW_UI__WEB__ENABLED=true \
    TZ=Europe/Paris

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:18789/health || exit 1

# Utilisateur non-root
USER openclaw

# Entrypoint avec tini (gestion propre des signaux)
ENTRYPOINT ["tini", "--", "/entrypoint.sh"]
CMD ["gateway"]
