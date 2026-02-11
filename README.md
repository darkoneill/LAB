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
│   ├── swarm.py            # Mode Swarm - sous-agents specialises (7 profils + routage dynamique)
│   ├── context.py          # Gestion du contexte avec compression
│   └── prompts/            # Templates de prompts (architecture AgentZero)
├── memory/                 # Systeme de memoire (architecture MemU)
│   ├── manager.py          # Gestionnaire autonome de memoire
│   ├── resource_layer.py   # Couche 1: Donnees brutes (jamais supprimees)
│   ├── item_layer.py       # Couche 2: Unites de memoire extraites
│   ├── category_layer.py   # Couche 3: Documents agregos lisibles
│   ├── retrieval.py        # Recherche hybride (keyword + semantique + contextuel)
│   └── evolution.py        # Auto-evolution, reflection + memoire episodique narrative
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

## Sublimation II - Ameliorations Avancees

### E. Agent Critique (Validation Hostile)

Un 6e profil d'agent dans le Swarm : le **Critic**, auditeur hostile et impartial.

**Role** : Apres la boucle Coder-Reviewer, le Critic valide la sortie finale en cherchant activement :
- Hallucinations et affirmations non fondees
- Erreurs logiques et contradictions
- Failles de securite (injections, fuites de donnees)
- Cas limites non geres
- Omissions du cahier des charges

**Flux** :
```
Coder -> Reviewer -> APPROVED -> Critic -> VALIDE? -> Retour utilisateur
                                    |
                                 REJETE -> Log warning + retour avec avertissement
```

### F. Mission Control (Visualisation des Traces)

Interface web dediee pour visualiser l'arbre de decision de l'agent en temps reel.

**Fonctionnalites** :
- Vue split : liste des traces (gauche) + arbre de spans (droite)
- Barre de stats : total, actives, duree moyenne, erreurs
- Icones et couleurs par type de span (LLM, outil, self-heal, etc.)
- Bandeau d'approbation temps reel via WebSocket
- Rafraichissement automatique a l'ouverture de l'onglet

### G. Memoire Episodique Narrative

"Consolidation Nocturne" : l'agent relit ses echanges et redige un resume de chapitre.

**Principe** : Apres une session longue (>= 5 echanges) ou une fois toutes les 6 heures, l'agent genere un resume narratif structure et l'ajoute a `PROJECT_MEMORY.md`.

**Structure du resume** :
1. Contexte - objectif de la session
2. Actions cles - decisions et actions prises
3. Resultats - ce qui a ete accompli
4. Lecons apprises - erreurs commises et corrigees
5. Pistes ouvertes - ce qui reste a faire

**Injection contextuelle** : Au demarrage d'une nouvelle session, les 3 derniers chapitres sont injectes dans le contexte systeme pour que l'agent se "souvienne" de ses sessions precedentes.

**Configuration** :
```yaml
memory:
  episodic:
    enabled: true
    file: "PROJECT_MEMORY.md"
    min_exchanges: 5
    min_interval_seconds: 21600  # 6 heures
    max_chapters_in_context: 3
```

### H. Mode Whisper (Approbation par Lots + Confiance Temporaire)

Amelioration UX de l'approbation Human-in-the-Loop.

**Approbation par lots** : Quand plusieurs operations sont en attente, un bouton "Tout autoriser" permet de toutes les approuver en un clic.

**Confiance temporaire** : Apres approbation, un selecteur permet d'accorder une confiance temporaire (5/15/30/60 min) pour que le meme outil soit auto-approuve pendant la duree choisie.

**Flux Whisper** :
```
3 operations en attente
       |
"Tout autoriser" + "Confiance 15 min"
       |
Auto-approbation pendant 15 min pour ces outils
```

**Configuration** :
```yaml
mcp:
  approval:
    trust_duration_minutes: 5   # Duree par defaut
```

**API** :

| Endpoint                       | Methode | Description                  |
|--------------------------------|---------|------------------------------|
| `/api/approvals/batch`         | POST    | Approbation par lots         |
| `/api/approvals/trusted`       | GET     | Outils avec confiance active |
| `/api/approvals/trust`         | POST    | Accorder confiance temp.     |
| `/api/approvals/trust`         | DELETE  | Revoquer confiance (`?tool_name=X&server_name=Y`) |

**WebSocket** : `{"type": "batch_approval", "approval_ids": [...], "approved": true, "trust_minutes": 15}`

---

## Sublimation III - Optimisations Millimétriques

### I. Self-Healing Environment-Aware (Contexte Sandbox)

Le prompt de Self-Healing est enrichi avec le contexte du conteneur (OS, version Python, packages installes).

**Principe** : Avant de demander au LLM de corriger le code, l'executor interroge le sandbox pour connaitre l'environnement. Si l'erreur est `ImportError: no module named pandas`, le LLM sait immediatement s'il doit utiliser une alternative stdlib (`csv`) ou generer un `pip install`.

**Exemple de contexte injecte** :
```
OS: Debian GNU/Linux 11 (bullseye)
Python: Python 3.11.9
Packages:
pip==24.0
setuptools==65.5.1
...
```

### J. Confiance Spatiale (Path-bound Trust)

La confiance temporaire (Whisper Mode) est desormais scopee par chemin de ressource.

**Principe** : Au lieu de faire confiance a `write_file` globalement, on peut restreindre la confiance a un prefixe de chemin specifique : `write_file` pour `/workspace/mon_projet/` uniquement.

**Resolution de confiance (plus specifique d'abord)** :
1. Confiance exacte : `server::write_file@/workspace/mon_projet/`
2. Confiance par prefixe : confiance sur `/workspace/` couvre `/workspace/foo/bar`
3. Confiance outil global : `server::write_file` (sans restriction de chemin)

### K. Terminal de Pensee (Thought Streaming)

Zone "Terminal de pensee" dans Mission Control qui affiche les tokens de reflexion du LLM en temps reel.

**Fonctionnalites** :
- Panneau collapsible en bas de Mission Control
- Texte gris a opacite reduite (style terminal)
- Badge compteur de chunks recus
- Auto-scroll vers le bas
- Support multi-agents avec labels `[coder]`, `[reviewer]`, etc.

**WebSocket** : `{"type": "thinking_stream", "text": "...", "agent": "coder", "new_turn": true}`

### L. Routage Dynamique de l'Essaim

Le Reviewer est desormais un routeur intelligent qui peut deleguer a des agents specialistes.

**Nouveau profil : Security Agent** - Expert OWASP/SANS, analyse les injections, fuites de secrets, SSRF, deserialisation, controle d'acces.

**Directives de routage** : Le Reviewer peut inclure `ROUTE:security` ou `ROUTE:tester` dans son feedback. L'orchestrateur parse ces directives et delegue automatiquement au specialiste.

**7 profils disponibles** :

| Role       | Acces Sandbox | Description                              |
|------------|---------------|------------------------------------------|
| `coder`    | Read/Write    | Expert Python strict, code executable    |
| `reviewer` | Read-Only     | Expert qualite + routeur intelligent     |
| `critic`   | Aucun         | Auditeur hostile et impartial            |
| `planner`  | Aucun         | Architecte, decompose les taches         |
| `tester`   | Read/Write    | Expert tests, ecrit des tests pytest     |
| `researcher`| Aucun        | Recherche et analyse d'information       |
| `security` | Read-Only     | Expert securite applicative (OWASP)      |

**Flux avec routage** :
```
Coder ecrit le code
       |
Reviewer analyse -> APPROVED? -> Critic -> Fin
       |
   ROUTE:security -> Security Agent -> Rapport
       |
   Feedback enrichi -> Coder corrige
```

## Sublimation IV - Resilience et Interaction (V7)

### M. Fading Memory (Anti-Token Overflow)

Compression automatique du feedback accumule dans la boucle Coder-Reviewer.

**Principe** : Apres l'iteration 2, si le feedback depasse 3000 caracteres, il est resume via un prompt LLM rapide avant d'etre passe au Coder. Cela evite la saturation de la fenetre de contexte et reduit les couts.

**Fallback** : Si la compression echoue, le feedback est tronque (les issues les plus recentes sont conservees).

### N. patch_file (Edition par Unified Diff)

Nouvel outil sandbox pour editer des fichiers sans les reecrire en entier.

**Format** : Le Coder peut envoyer des edits search/replace :
```json
{
  "tool": "patch_file",
  "path": "src/app.py",
  "edits": [
    {"search": "x = 1", "replace": "x = 2"},
    {"search": "def old_func():", "replace": "def new_func():"}
  ]
}
```

**Avantage** : Evite les erreurs de troncature sur les gros fichiers (> max_tokens). Meme approche que Aider/Cursor.

### O. Human Hinting (Murmurer a l'Agent)

Injection de contexte humain en temps reel pendant l'execution de l'essaim.

**Fonctionnalite** : Champ "Murmurer a l'agent..." sous le Terminal de Pensee. Le conseil est envoye via WebSocket et injecte comme `[MESSAGE URGENT DE L'UTILISATEUR]` lors de la prochaine iteration du Coder.

**WebSocket** : `{"type": "human_hint", "text": "Utilise csv au lieu de pandas"}`

### P. Salle de Reunion (Swarm Visualization)

Representation graphique de l'essaim d'agents dans Mission Control.

**Fonctionnalite** : Barre horizontale avec des pastilles colorees pour chaque role d'agent. Les pastilles s'illuminent (pulse) quand l'agent correspondant est actif.

**Evenements WebSocket** :
- `agent_spawned` : la pastille s'active
- `agent_completed` / `agent_failed` : la pastille s'eteint

**Roles affiches** : Plan (bleu), Code (vert), Review (jaune), Secu (rouge), Test (teal), Critic (rose), Rech (violet).

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
| `/api/approvals/batch`            | POST    | Approbation par lots (Whisper) |
| `/api/approvals/trusted`          | GET     | Outils avec confiance temp.    |
| `/api/approvals/trust`            | POST    | Accorder confiance temporaire  |
| `/api/approvals/trust`            | DELETE  | Revoquer confiance (query params) |
| `/ws/{client_id}`                 | WS      | WebSocket temps reel           |

## Licence

MIT
