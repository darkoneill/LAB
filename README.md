# OpenClaw

**NexusMind v1.0.0** - Assistant IA Autonome

Clone abouti combinant les meilleures fonctionnalites de :
- **OpenClaw** : Architecture gateway-centree, skills modulaires, execution systeme
- **MemU** : Memoire persistante en 3 couches, auto-organisation, evolution
- **AgentZero** : Delegation hierarchique, creation d'outils, architecture prompt-driven

## Demarrage Rapide

```bash
# Installer les dependances
pip install -r requirements.txt

# Lancer (le wizard se lance automatiquement au premier demarrage)
python run.py

# Ou specifiquement :
python run.py terminal    # Terminal seulement
python run.py gateway     # API Gateway seulement
python run.py wizard      # Relancer le wizard
python run.py both        # Terminal + Gateway (defaut)
```

## Architecture

```
openclaw/
├── main.py                 # Point d'entree principal
├── setup_wizard.py         # Wizard d'installation interactif
├── config/
│   ├── default.yaml        # Configuration par defaut
│   ├── user.yaml           # Configuration utilisateur (generee)
│   └── settings.py         # Gestionnaire de configuration
├── gateway/                # Gateway API (FastAPI + WebSocket)
│   ├── server.py           # Serveur HTTP/WS avec SSE streaming
│   ├── router.py           # Routage intelligent avec failover
│   └── middleware.py       # Securite, rate limiting, cache semantique
├── agent/                  # Cerveau de l'IA
│   ├── brain.py            # Moteur de raisonnement multi-modele
│   ├── orchestrator.py     # Orchestration multi-agent hierarchique
│   ├── context.py          # Gestion du contexte avec compression
│   └── prompts/            # Templates de prompts (architecture AgentZero)
├── memory/                 # Systeme de memoire (architecture MemU)
│   ├── manager.py          # Gestionnaire autonome de memoire
│   ├── resource_layer.py   # Couche 1: Donnees brutes (jamais supprimees)
│   ├── item_layer.py       # Couche 2: Unites de memoire extraites
│   ├── category_layer.py   # Couche 3: Documents agregos lisibles
│   ├── retrieval.py        # Recherche hybride (keyword + semantique + contextuel)
│   └── evolution.py        # Auto-evolution et reflection
├── skills/                 # Systeme de skills modulaires
│   ├── base.py             # Classe de base pour les skills
│   ├── loader.py           # Decouverte et chargement dynamique
│   ├── router.py           # Routage d'intention vers les skills
│   └── builtin/            # Skills integres
│       ├── file_manager/   # Operations fichiers
│       ├── code_executor/  # Execution de code multi-langage
│       ├── web_search/     # Recherche web (DuckDuckGo)
│       └── system_info/    # Informations systeme
├── tools/                  # Outils de base (philosophie AgentZero)
│   └── executor.py         # 4 outils par defaut: shell, read, write, search
└── ui/
    ├── terminal.py         # Interface terminal Rich
    └── web/                # Interface web
        ├── app.py          # Application FastAPI
        ├── templates/      # Templates HTML
        └── static/         # CSS + JavaScript
```

## Fonctionnalites

### Gateway (inspire des meilleures pratiques 2025-2026)
- API REST + WebSocket temps reel
- Streaming SSE token par token
- Cache semantique (reponses similaires cachees)
- Rate limiting base sur les tokens
- Failover automatique entre providers
- Securite : detection d'injection, filtrage PII, validation de contenu
- Gestion de sessions avec historique

### Agent (inspire de AgentZero)
- Architecture **prompt-driven** : tout le comportement est defini dans des fichiers editables
- Support **multi-modeles** : Anthropic, OpenAI, Ollama, custom
- **Delegation hierarchique** : creation d'agents specialises pour les sous-taches
- **4 outils de base** : shell, read_file, write_file, search_files
- Compression dynamique du contexte
- Failover automatique entre providers

### Memoire (inspiree de MemU)
- **3 couches** :
  - Resource Layer : donnees brutes, jamais supprimees
  - Item Layer : unites de memoire minimales et independantes
  - Category Layer : documents markdown agregos et lisibles
- Recherche **hybride** : keyword (TF-IDF) + contextuelle + semantique
- **Auto-organisation** : categorisation automatique des souvenirs
- **Evolution** : reflexion periodique et generation d'insights
- **Oubli gracieux** : les memoires non-accedees perdent en priorite, mais ne sont jamais effacees

### Skills
- Systeme modulaire avec decouverte automatique
- Chaque skill est un dossier avec `SKILL.md` (metadonnees YAML) + `skill.py`
- Routage d'intention automatique
- Skills integres : fichiers, code, web, systeme
- Creation de skills a la volee par l'agent

### Interface
- **Terminal** : interface Rich avec markdown, tableaux, spinners
- **Web** : SPA avec dark theme, WebSocket temps reel, chat + memoire + config
- **Wizard** : installation guidee et conversationnelle

## Configuration

Variables d'environnement supportees :
```bash
ANTHROPIC_API_KEY=sk-ant-...     # Cle API Anthropic
OPENAI_API_KEY=sk-...            # Cle API OpenAI
OPENCLAW_GATEWAY__PORT=18789     # Override du port
OPENCLAW_LOGGING__LEVEL=DEBUG    # Niveau de log
```

## Commandes Terminal

| Commande   | Description                          |
|------------|--------------------------------------|
| `/help`    | Afficher l'aide                      |
| `/status`  | Statut du systeme                    |
| `/config`  | Voir/modifier la configuration       |
| `/memory`  | Explorer la memoire                  |
| `/skills`  | Lister les skills                    |
| `/tools`   | Lister les outils                    |
| `/clear`   | Effacer l'ecran                      |
| `/reset`   | Reinitialiser la session             |
| `/wizard`  | Relancer le wizard                   |
| `/quit`    | Quitter                              |

## API Gateway

| Endpoint                          | Methode | Description                    |
|-----------------------------------|---------|--------------------------------|
| `/health`                         | GET     | Sante du serveur               |
| `/api/info`                       | GET     | Infos sur le systeme           |
| `/api/chat`                       | POST    | Chat (streaming ou non)        |
| `/api/chat/simple`                | POST    | Chat simplifie                 |
| `/api/sessions`                   | GET     | Lister les sessions            |
| `/api/sessions/{id}/history`      | GET     | Historique d'une session       |
| `/api/memory/search?query=...`    | GET     | Recherche memoire              |
| `/api/memory/categories`          | GET     | Categories memoire             |
| `/api/skills`                     | GET     | Lister les skills              |
| `/api/models`                     | GET     | Modeles disponibles            |
| `/api/config`                     | GET/PUT | Configuration                  |
| `/ws/{client_id}`                 | WS      | WebSocket temps reel           |

## Licence

MIT
