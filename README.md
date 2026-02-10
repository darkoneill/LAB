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
│   ├── middleware.py        # Securite, rate limiting, cache semantique
│   └── approval.py         # Middleware d'approbation Human-in-the-Loop (MCP)
├── agent/                  # Cerveau de l'IA
│   ├── brain.py            # Moteur de raisonnement multi-modele
│   ├── orchestrator.py     # Orchestration multi-agent hierarchique
│   ├── swarm.py            # Mode Swarm - sous-agents specialises (Coder/Reviewer/Tester)
│   ├── context.py          # Gestion du contexte avec compression
│   └── prompts/            # Templates de prompts (architecture AgentZero)
├── memory/                 # Systeme de memoire (architecture MemU)
│   ├── manager.py          # Gestionnaire autonome de memoire
│   ├── resource_layer.py   # Couche 1: Donnees brutes (jamais supprimees)
│   ├── item_layer.py       # Couche 2: Unites de memoire extraites
│   ├── category_layer.py   # Couche 3: Documents agregos lisibles
│   ├── retrieval.py        # Recherche hybride (keyword + semantique + contextuel)
│   └── evolution.py        # Auto-evolution et reflection
├── tracing/                # Observabilite et Black Box Recorder
│   ├── __init__.py         # Export du TraceRecorder
│   └── recorder.py         # Enregistrement structure de chaque etape du pipeline
├── sandbox/                # Execution isolee
│   ├── container.py        # Gestion du cycle de vie des conteneurs Docker
│   └── executor.py         # Executeur sandbox avec Self-Healing Code Loop
├── mcp/                    # Model Context Protocol
│   ├── client.py           # Client MCP (stdio/SSE)
│   └── registry.py         # Registre multi-serveurs avec approbation
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

---

## Fonctionnalites Avancees (Sublimation)

### A. Self-Healing Code Loop (Auto-Correction)

L'agent peut debugger son propre code sans intervention humaine.

**Principe** : Quand du code Python echoue dans le sandbox, le systeme intercepte l'erreur, la renvoie au LLM avec le code original, et le LLM produit une version corrigee. Max 3 tentatives.

**Erreurs gerees** : `ModuleNotFoundError`, `SyntaxError`, `TypeError`, `NameError`, `ImportError`, `AttributeError`, `KeyError`, `IndexError`, `ValueError`, `FileNotFoundError`, `ZeroDivisionError`, `IndentationError`.

**Configuration** (`config/default.yaml`) :
```yaml
sandbox:
  self_healing:
    enabled: true
    max_attempts: 3
```

**Flux** :
```
Code Python -> Sandbox -> Erreur
                           |
                    LLM: "Corrige ce code"
                           |
                    Code corrige -> Sandbox -> Succes (ou retry)
```

### B. Observabilite / Black Box Recorder

Chaque etape du pipeline est enregistree dans une trace structuree pour le debug et le replay.

**Spans enregistres** : `REQUEST`, `RETRIEVAL`, `LLM_CALL`, `TOOL_EXEC`, `SELF_HEAL`, `DELEGATION`, `MCP_CALL`, `APPROVAL`, `RESPONSE`.

**Stockage** :
- Ring buffer en memoire (500 traces, acces O(1) par index)
- Persistance JSON sur disque pour l'historique long terme

**Configuration** :
```yaml
tracing:
  enabled: true
  max_traces: 500
  persist: true
  store_path: "logs/traces"
```

**API** :

| Endpoint                      | Methode | Description              |
|-------------------------------|---------|--------------------------|
| `/api/traces`                 | GET     | Liste des traces         |
| `/api/traces/stats`           | GET     | Statistiques (avg, p95)  |
| `/api/traces/search/{query}`  | GET     | Recherche par contenu    |
| `/api/traces/{trace_id}`      | GET     | Detail d'une trace       |

### C. Mode Swarm (Essaim de Sous-Agents)

L'orchestrateur peut creer des agents specialises avec des profils systeme distincts.

**5 profils disponibles** :

| Role       | Acces Sandbox | Description                              |
|------------|---------------|------------------------------------------|
| `coder`    | Read/Write    | Expert Python strict, code executable    |
| `reviewer` | Read-Only     | Expert securite, cherche les failles     |
| `planner`  | Aucun         | Architecte, decompose les taches         |
| `tester`   | Read/Write    | Expert tests, ecrit des tests pytest     |
| `researcher`| Aucun        | Recherche et analyse d'information       |

**Boucle Coder-Reviewer** :
```
Coder ecrit le code
       |
Reviewer analyse -> APPROVED? -> Fin
       |
   Corrections necessaires
       |
Coder corrige (iteration N+1)
```

**Configuration** :
```yaml
agent:
  swarm:
    enabled: true
    max_iterations: 3
```

**API** :

| Endpoint               | Methode | Description                |
|------------------------|---------|----------------------------|
| `/api/swarm/profiles`  | GET     | Profils d'agents dispo     |
| `/api/swarm/execute`   | POST    | Lancer un essaim           |

### D. Interface MCP Human-in-the-Loop

Les outils MCP sont classes par niveau de risque. Les operations sensibles necessitent l'approbation de l'utilisateur.

**Classification automatique** :

| Niveau     | Operations                                  | Comportement       |
|------------|---------------------------------------------|---------------------|
| `safe`     | `get_*`, `list_*`, `read_*`, `search_*`    | Auto-approuve       |
| `sensitive`| `create_*`, `write_*`, `update_*`, `send_*` | Approbation requise |
| `critical` | `delete_*`, `destroy_*`, `drop_*`, `kill_*` | Approbation requise |

**Flux** :
```
Agent appelle un outil MCP sensible
       |
Middleware intercepte -> Notification WebSocket vers l'UI
       |
Utilisateur: "Autoriser" ou "Refuser"
       |
Execution ou annulation
```

**Configuration** :
```yaml
mcp:
  approval:
    enabled: true
    timeout_seconds: 120
    auto_approve_safe: true
    tool_overrides: {}    # ex: { "my_tool": "safe" }
```

**API** :

| Endpoint                       | Methode | Description                  |
|--------------------------------|---------|------------------------------|
| `/api/approvals/pending`       | GET     | Approbations en attente      |
| `/api/approvals/{id}`          | POST    | Approuver/refuser            |
| `/api/approvals/history`       | GET     | Historique des decisions      |

**WebSocket** : Envoyer `{"type": "approval_response", "approval_id": "...", "approved": true}` pour repondre.

---

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
| `/api/traces`                     | GET     | Traces du pipeline             |
| `/api/traces/stats`               | GET     | Statistiques de tracing        |
| `/api/traces/{id}`                | GET     | Detail d'une trace             |
| `/api/swarm/profiles`             | GET     | Profils d'agents swarm         |
| `/api/swarm/execute`              | POST    | Executer un essaim             |
| `/api/approvals/pending`          | GET     | Approbations en attente        |
| `/api/approvals/{id}`             | POST    | Approuver/refuser              |
| `/api/approvals/history`          | GET     | Historique approbations        |
| `/ws/{client_id}`                 | WS      | WebSocket temps reel           |

## Licence

MIT
