#!/bin/bash
# ============================================================
# OpenClaw NexusMind - Script d'installation VPS
# Compatible : Ubuntu 22.04+ / Debian 12+ (Hostinger KVM)
#
# Usage:
#   chmod +x docker/setup-vps.sh
#   sudo ./docker/setup-vps.sh
# ============================================================

set -e

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}"
echo "  ___                    ____ _                "
echo " / _ \ _ __   ___ _ __ / ___| | __ ___      __"
echo "| | | | '_ \ / _ \ '_ \ |   | |/ _\` \ \ /\ / /"
echo "| |_| | |_) |  __/ | | | |___| | (_| |\ V  V / "
echo " \___/| .__/ \___|_| |_|\____|_|\__,_| \_/\_/  "
echo "      |_|             Setup VPS Hostinger"
echo -e "${NC}"
echo ""

# ── Verification root ─────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}[erreur]${NC} Ce script doit etre lance en root (sudo)"
    exit 1
fi

# ── Detection OS ──────────────────────────────────────────────
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
else
    OS=$(uname -s)
    VER=$(uname -r)
fi
echo -e "${GREEN}[info]${NC} Systeme detecte: $OS $VER"

# ── 1. Mise a jour systeme ────────────────────────────────────
echo -e "\n${CYAN}[1/6]${NC} Mise a jour du systeme..."
apt-get update -qq
apt-get upgrade -y -qq

# ── 2. Installation Docker ────────────────────────────────────
echo -e "\n${CYAN}[2/6]${NC} Installation de Docker..."
if command -v docker &> /dev/null; then
    echo -e "${GREEN}[ok]${NC} Docker deja installe ($(docker --version))"
else
    apt-get install -y -qq ca-certificates curl gnupg
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
        tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    systemctl enable docker
    systemctl start docker
    echo -e "${GREEN}[ok]${NC} Docker installe"
fi

# ── 3. Configuration firewall ─────────────────────────────────
echo -e "\n${CYAN}[3/6]${NC} Configuration du firewall..."
if command -v ufw &> /dev/null; then
    ufw allow 22/tcp    # SSH
    ufw allow 80/tcp    # HTTP
    ufw allow 443/tcp   # HTTPS
    ufw --force enable
    echo -e "${GREEN}[ok]${NC} Firewall configure (22, 80, 443)"
else
    echo -e "${YELLOW}[skip]${NC} ufw non disponible, configurer manuellement"
fi

# ── 4. Configuration swap (pour petit VPS) ────────────────────
echo -e "\n${CYAN}[4/6]${NC} Configuration swap..."
TOTAL_RAM=$(free -m | awk '/^Mem:/{print $2}')
echo -e "${GREEN}[info]${NC} RAM detectee: ${TOTAL_RAM}MB"

if [ ! -f /swapfile ] && [ "$TOTAL_RAM" -lt 2048 ]; then
    SWAP_SIZE=$((TOTAL_RAM * 2))
    echo -e "${GREEN}[info]${NC} Creation swap de ${SWAP_SIZE}MB..."
    fallocate -l ${SWAP_SIZE}M /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    # Optimiser swappiness pour VPS
    sysctl vm.swappiness=10
    echo 'vm.swappiness=10' >> /etc/sysctl.conf
    echo -e "${GREEN}[ok]${NC} Swap de ${SWAP_SIZE}MB active"
else
    echo -e "${GREEN}[ok]${NC} Swap OK ou RAM suffisante"
fi

# ── 5. Preparation projet ────────────────────────────────────
echo -e "\n${CYAN}[5/6]${NC} Preparation du projet..."
INSTALL_DIR="/opt/openclaw"

if [ ! -d "$INSTALL_DIR" ]; then
    mkdir -p "$INSTALL_DIR"
    echo -e "${GREEN}[info]${NC} Repertoire cree: $INSTALL_DIR"
fi

# Copier les fichiers si on est dans le repo
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$SCRIPT_DIR/Dockerfile" ]; then
    cp -r "$SCRIPT_DIR"/* "$INSTALL_DIR/"
    cp "$SCRIPT_DIR"/.env.example "$INSTALL_DIR/.env.example"
    cp "$SCRIPT_DIR"/.dockerignore "$INSTALL_DIR/.dockerignore" 2>/dev/null || true
    echo -e "${GREEN}[ok]${NC} Fichiers copies dans $INSTALL_DIR"
fi

# Creer .env si absent
if [ ! -f "$INSTALL_DIR/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    echo -e "${YELLOW}[action]${NC} Fichier .env cree - EDITER AVANT DE LANCER :"
    echo -e "          ${CYAN}nano $INSTALL_DIR/.env${NC}"
fi

# ── 6. Recommandations memoire selon RAM ──────────────────────
echo -e "\n${CYAN}[6/6]${NC} Recommandations..."

if [ "$TOTAL_RAM" -lt 1024 ]; then
    RECOMMENDED_LIMIT="256M"
    echo -e "${YELLOW}[warning]${NC} RAM faible (${TOTAL_RAM}MB)"
    echo "  -> Utiliser MEMORY_LIMIT=256M dans .env"
    echo "  -> Considerer Ollama avec un petit modele ou API cloud"
elif [ "$TOTAL_RAM" -lt 2048 ]; then
    RECOMMENDED_LIMIT="512M"
elif [ "$TOTAL_RAM" -lt 4096 ]; then
    RECOMMENDED_LIMIT="1G"
else
    RECOMMENDED_LIMIT="2G"
fi

# Appliquer la recommandation dans .env
if [ -f "$INSTALL_DIR/.env" ]; then
    sed -i "s/MEMORY_LIMIT=.*/MEMORY_LIMIT=$RECOMMENDED_LIMIT/" "$INSTALL_DIR/.env"
fi

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Installation terminee !${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo -e "  Prochaines etapes :"
echo ""
echo -e "  1. Configurer les cles API :"
echo -e "     ${CYAN}nano $INSTALL_DIR/.env${NC}"
echo ""
echo -e "  2. Lancer OpenClaw :"
echo -e "     ${CYAN}cd $INSTALL_DIR${NC}"
echo -e "     ${CYAN}docker compose up -d${NC}"
echo ""
echo -e "  3. Verifier le statut :"
echo -e "     ${CYAN}docker compose ps${NC}"
echo -e "     ${CYAN}docker compose logs -f app${NC}"
echo ""
echo -e "  4. (Optionnel) Configurer SSL :"
echo -e "     ${CYAN}docker compose run --rm certbot certonly --webroot -w /var/www/certbot -d VOTRE_DOMAINE --email VOTRE_EMAIL --agree-tos${NC}"
echo ""
echo -e "  L'interface web sera disponible sur :"
echo -e "     ${CYAN}http://IP_DU_VPS${NC}"
echo ""
