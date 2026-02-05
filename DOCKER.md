# OpenClaw - Deploiement Docker (VPS Hostinger)

## Prerequis

- VPS Ubuntu 22.04+ ou Debian 12+ (Hostinger KVM recommande)
- Minimum 1 Go RAM (2 Go+ recommande)
- Au moins une cle API (Anthropic ou OpenAI)
- Un domaine pointe vers l'IP du VPS (optionnel, pour SSL)

## Installation rapide

### 1. Cloner le projet sur le VPS

```bash
ssh root@IP_DU_VPS
git clone https://github.com/darkoneill/LAB.git /opt/openclaw
cd /opt/openclaw
```

### 2. Script d'installation automatique

```bash
chmod +x docker/setup-vps.sh
sudo ./docker/setup-vps.sh
```

Ce script :
- Met a jour le systeme
- Installe Docker + Docker Compose
- Configure le firewall (ports 22, 80, 443)
- Configure le swap si RAM < 2 Go
- Detecte la RAM et ajuste `MEMORY_LIMIT` dans `.env`
- Copie les fichiers dans `/opt/openclaw`

### 3. Configurer les cles API

```bash
nano .env
```

Remplir au minimum :
```env
ANTHROPIC_API_KEY=sk-ant-api03-VOTRE_CLE_ICI
```

### 4. Lancer

```bash
docker compose up -d
```

L'interface est disponible sur `http://IP_DU_VPS`.

---

## Variantes de deploiement

### Version complete (avec Nginx + SSL)

```bash
docker compose up -d
```

Inclut : app + nginx reverse proxy + certbot (SSL) + watchtower (optionnel).

### Version legere (sans Nginx)

```bash
docker compose -f docker/docker-compose.light.yml up -d
```

Acces direct au port 18789. Utile si un reverse proxy existe deja ou pour du dev.

### Avec Watchtower (mise a jour auto)

```bash
docker compose --profile auto-update up -d
```

---

## Configuration SSL (Let's Encrypt)

### 1. Pointer le domaine

Dans le panel DNS Hostinger, creer un enregistrement A :
```
Type: A
Nom:  @  (ou sous-domaine)
IP:   IP_DU_VPS
TTL:  3600
```

### 2. Configurer le domaine

```bash
nano .env
```
```env
DOMAIN=openclaw.mondomaine.com
CERTBOT_EMAIL=email@example.com
```

### 3. Obtenir le certificat

```bash
docker compose up -d nginx

docker compose run --rm certbot certonly \
    --webroot -w /var/www/certbot \
    -d openclaw.mondomaine.com \
    --email email@example.com \
    --agree-tos \
    --no-eff-email
```

### 4. Activer HTTPS dans Nginx

Editer `docker/nginx/conf.d/default.conf` :
- Decommenter le block `server 443`
- Remplacer `YOUR_DOMAIN` par votre domaine
- Decommenter la redirection HTTP -> HTTPS dans le block `server 80`

```bash
docker compose restart nginx
```

### 5. Renouvellement automatique

Le service `certbot` dans le compose renouvelle automatiquement.
Verification manuelle :
```bash
docker compose run --rm certbot renew --dry-run
```

---

## Commandes utiles

```bash
# Statut des services
docker compose ps

# Logs en temps reel
docker compose logs -f app
docker compose logs -f nginx

# Redemarrer l'application
docker compose restart app

# Rebuild apres mise a jour du code
docker compose build --no-cache app
docker compose up -d app

# Arreter tout
docker compose down

# Arreter et supprimer les volumes (ATTENTION: perte de donnees)
docker compose down -v

# Entrer dans le container
docker compose exec app bash

# Verifier la sante
curl http://localhost:18789/health

# Voir l'utilisation des ressources
docker stats openclaw
```

---

## Gestion des donnees

### Volumes Docker

| Volume | Contenu | Critique |
|--------|---------|----------|
| `openclaw_memory` | Memoire persistante (3 couches) | Oui |
| `openclaw_config` | Configuration utilisateur | Oui |
| `openclaw_logs` | Fichiers de log | Non |
| `openclaw_skills` | Skills personnalises | Selon usage |

### Sauvegarde

```bash
# Sauvegarder les donnees
docker compose exec app tar czf /tmp/backup.tar.gz /data
docker compose cp app:/tmp/backup.tar.gz ./backup-$(date +%Y%m%d).tar.gz

# Restaurer
docker compose cp ./backup-20260205.tar.gz app:/tmp/backup.tar.gz
docker compose exec app tar xzf /tmp/backup.tar.gz -C /
```

### Sauvegarde automatique (cron)

```bash
crontab -e
```
```cron
# Sauvegarde quotidienne a 3h du matin
0 3 * * * cd /opt/openclaw && docker compose exec -T app tar czf /tmp/backup.tar.gz /data && docker compose cp app:/tmp/backup.tar.gz /opt/openclaw/backups/backup-$(date +\%Y\%m\%d).tar.gz && find /opt/openclaw/backups -mtime +7 -delete
```

---

## Ressources selon plan Hostinger

| Plan VPS | RAM | MEMORY_LIMIT | CPU_LIMIT | Notes |
|----------|-----|-------------|-----------|-------|
| KVM 1 | 1 Go | 256M | 1.0 | API cloud uniquement, pas d'Ollama |
| KVM 2 | 2 Go | 512M | 1.0 | Correct pour usage normal |
| KVM 4 | 4 Go | 1G | 2.0 | Confortable |
| KVM 8 | 8 Go | 2G | 4.0 | Peut faire tourner Ollama avec petit modele |

---

## Monitoring

### Health check

L'endpoint `/health` retourne :
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "uptime_seconds": 3600,
    "active_sessions": 2,
    "memory_usage_mb": 85.4
}
```

### Surveiller avec un outil externe

```bash
# Simple cron check
*/5 * * * * curl -sf http://localhost:18789/health || docker compose -f /opt/openclaw/docker-compose.yml restart app
```

---

## Depannage

### Le container ne demarre pas

```bash
docker compose logs app            # Voir les erreurs
docker compose exec app cat /data/logs/openclaw.log
```

### Erreur "out of memory"

Reduire `MEMORY_LIMIT` dans `.env` ou augmenter le swap :
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Nginx retourne 502 Bad Gateway

```bash
docker compose ps app              # Verifier que l'app tourne
docker compose logs app            # Verifier les logs
docker compose restart app nginx   # Redemarrer
```

### Reponses LLM lentes

- Verifier la latence API : `curl -w "%{time_total}\n" -o /dev/null -s https://api.anthropic.com`
- Activer le cache semantique dans la config
- Utiliser un modele plus rapide (Haiku, GPT-4o-mini)

### Reset complet

```bash
docker compose down -v
rm -rf /opt/openclaw/.env
cp .env.example .env
nano .env
docker compose up -d
```
