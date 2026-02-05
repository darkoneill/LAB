#!/bin/bash
set -e

# ============================================================
# OpenClaw - Docker Entrypoint
# ============================================================

APP_DIR="/app"
DATA_DIR="${OPENCLAW_DATA_DIR:-/data}"

# Couleurs
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ___                    ____ _                "
echo " / _ \ _ __   ___ _ __ / ___| | __ ___      __"
echo "| | | | '_ \ / _ \ '_ \ |   | |/ _\` \ \ /\ / /"
echo "| |_| | |_) |  __/ | | | |___| | (_| |\ V  V / "
echo " \___/| .__/ \___|_| |_|\____|_|\__,_| \_/\_/  "
echo "      |_|                    NexusMind v1.0.0"
echo -e "${NC}"

# ── Initialisation des repertoires ────────────────────────────
echo -e "${GREEN}[init]${NC} Verification des repertoires..."
mkdir -p "${DATA_DIR}/memory" \
         "${DATA_DIR}/config" \
         "${DATA_DIR}/logs" \
         "${DATA_DIR}/skills"

# ── Copier la config par defaut si absente ────────────────────
if [ ! -f "${DATA_DIR}/config/user.yaml" ]; then
    echo -e "${YELLOW}[init]${NC} Premier demarrage detecte"

    # Generer user.yaml depuis les variables d'environnement
    cat > "${DATA_DIR}/config/user.yaml" <<YAML
# OpenClaw - Configuration utilisateur (generee automatiquement)
# Modifiable via l'API /api/config ou directement ici

app:
  name: "${OPENCLAW_APP_NAME:-OpenClaw}"
  debug: ${OPENCLAW_DEBUG:-false}

gateway:
  host: "${OPENCLAW_GATEWAY__HOST:-0.0.0.0}"
  port: ${OPENCLAW_GATEWAY__PORT:-18789}
  security:
    api_key_required: ${OPENCLAW_API_KEY_REQUIRED:-false}

providers:
  anthropic:
    enabled: ${ANTHROPIC_ENABLED:-true}
    api_key: "${ANTHROPIC_API_KEY:-}"
    default_model: "${ANTHROPIC_MODEL:-claude-sonnet-4-20250514}"
  openai:
    enabled: ${OPENAI_ENABLED:-false}
    api_key: "${OPENAI_API_KEY:-}"
    default_model: "${OPENAI_MODEL:-gpt-4o}"
  ollama:
    enabled: ${OLLAMA_ENABLED:-false}
    base_url: "${OLLAMA_URL:-http://ollama:11434}"
    default_model: "${OLLAMA_MODEL:-llama3.2}"

memory:
  enabled: true
  store_path: "${DATA_DIR}/memory"

logging:
  level: "${OPENCLAW_LOGGING__LEVEL:-INFO}"
  file: "${DATA_DIR}/logs/openclaw.log"

ui:
  web:
    enabled: true
YAML

    echo -e "${GREEN}[init]${NC} Configuration generee dans ${DATA_DIR}/config/user.yaml"
else
    echo -e "${GREEN}[init]${NC} Configuration existante detectee"
fi

# ── Lien symbolique config ────────────────────────────────────
if [ -f "${DATA_DIR}/config/user.yaml" ] && [ ! -f "${APP_DIR}/openclaw/config/user.yaml" ]; then
    ln -sf "${DATA_DIR}/config/user.yaml" "${APP_DIR}/openclaw/config/user.yaml" 2>/dev/null || true
fi

# ── Lien symbolique skills custom ────────────────────────────
if [ -d "${DATA_DIR}/skills" ] && [ ! -d "${APP_DIR}/openclaw/skills/custom" ]; then
    ln -sf "${DATA_DIR}/skills" "${APP_DIR}/openclaw/skills/custom" 2>/dev/null || true
fi

# ── Lancement ─────────────────────────────────────────────────
MODE="${1:-gateway}"

case "$MODE" in
    gateway)
        echo -e "${GREEN}[start]${NC} Mode: Gateway API (port ${OPENCLAW_GATEWAY__PORT:-18789})"
        exec python run.py gateway --no-wizard
        ;;
    terminal)
        echo -e "${GREEN}[start]${NC} Mode: Terminal interactif"
        exec python run.py terminal
        ;;
    both)
        echo -e "${GREEN}[start]${NC} Mode: Gateway + Terminal"
        exec python run.py both --no-wizard
        ;;
    wizard)
        echo -e "${GREEN}[start]${NC} Mode: Setup Wizard"
        exec python run.py wizard
        ;;
    *)
        echo -e "${GREEN}[start]${NC} Commande custom: $@"
        exec "$@"
        ;;
esac
