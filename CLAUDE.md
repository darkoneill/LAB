# CLAUDE.md — NexusMind v1.0.0 (OpenClaw Python Clone)

## Projet

NexusMind est un assistant IA autonome en Python (~10K lignes) fusionnant les architectures d'OpenClaw (gateway), MemU (mémoire 3 couches) et AgentZero (délégation hiérarchique). Le repo est dans `openclaw/`.

## Stack technique

- **Langage** : Python 3.11+, 100% async (asyncio)
- **Framework web** : FastAPI + Uvicorn + WebSocket + SSE
- **LLM** : Anthropic SDK (natif) + OpenAI-compatible (Ollama, etc.)
- **Mémoire** : ChromaDB (vector), scikit-learn TF-IDF, sentence-transformers
- **Sandbox** : Docker SDK (isolation des commandes dangereuses)
- **UI** : Rich (terminal) + Jinja2/vanilla JS (web)
- **Tests** : pytest + pytest-asyncio
- **Config** : YAML (pydantic-settings), variables d'env `OPENCLAW_*`

## Architecture clé

```
openclaw/
├── agent/
│   ├── brain.py          # Moteur LLM multi-provider + native tool calling
│   ├── orchestrator.py   # Délégation hiérarchique (analyse → sous-agents → agrégation)
│   ├── swarm.py          # Multi-agent spécialisé (Coder→Reviewer→Critic pipeline)
│   ├── scheduler.py      # Cron agentic (heartbeats planifiés)
│   └── context.py        # Compression de contexte
├── memory/
│   ├── manager.py        # Gestionnaire autonome 3 couches
│   ├── resource_layer.py # Couche 1 : données brutes (JAMAIS supprimées)
│   ├── item_layer.py     # Couche 2 : unités extraites + scoring significance
│   ├── category_layer.py # Couche 3 : documents markdown agrégés + évolution
│   ├── retrieval.py      # Recherche hybride (TF-IDF + contextuel + sémantique)
│   ├── vector_store.py   # ChromaDB wrapper (lazy-loaded)
│   └── evolution.py      # Réflexion périodique + insights
├── gateway/
│   ├── server.py         # FastAPI : REST + WebSocket + SSE + sessions
│   ├── router.py         # Routage multi-provider avec failover + health tracking
│   ├── middleware.py      # Rate limiting, sécurité, cache sémantique
│   └── approval.py       # Système d'approbation des actions dangereuses
├── sandbox/
│   ├── executor.py       # Routing auto sandbox + self-healing code loop
│   └── container.py      # Gestion conteneurs Docker isolés
├── mcp/
│   ├── client.py         # Client MCP (stdio + SSE)
│   ├── server.py         # Serveur MCP exposant les tools NexusMind
│   └── registry.py       # Registre de serveurs MCP
├── skills/
│   ├── base.py           # ABC avec métadonnées YAML (SKILL.md)
│   ├── loader.py         # Découverte et chargement dynamique
│   ├── router.py         # Routage d'intention (tags + keywords + scoring)
│   └── builtin/          # file_manager, code_executor, web_search, system_info
├── tools/
│   ├── executor.py       # 4 outils : shell, read_file, write_file, search_files
│   └── repo_map.py       # Squelette AST du repo pour le Swarm
├── tracing/
│   └── recorder.py       # Observabilité : spans, latence, coûts tokens
├── ui/
│   ├── terminal.py       # Interface Rich (markdown, tableaux, spinners)
│   └── web/              # SPA : FastAPI templates + JS + dark theme
├── config/
│   ├── settings.py       # Singleton Settings (YAML + env overrides)
│   └── default.yaml      # Configuration par défaut
├── main.py               # Point d'entrée (terminal / gateway / both / wizard)
└── setup_wizard.py       # Installation guidée interactive
```

## Conventions de code

- Tout est **async/await**. Pas de code bloquant dans la boucle événementielle.
- Les imports sont **relatifs** dans le package (`from .resource_layer import ResourceLayer`).
- Logger par module : `logger = logging.getLogger("openclaw.module.submodule")`
- Les settings s'accèdent via `get_settings().get("section.key", default)`.
- Les classes utilisent des **dataclasses** ou **pydantic BaseModel** pour les structures.
- Docstrings en anglais, commentaires en français acceptés.
- Type hints obligatoires sur les signatures publiques.
- Les commandes terminal commencent par `/` (ex: `/help`, `/memory`, `/status`).

## Patterns importants

### Tool Calling natif
`brain.py` envoie les tool definitions au LLM via le paramètre `tools=` (Anthropic et OpenAI format). Les tool_use blocks de la réponse sont parsés et exécutés par `_execute_tools()`. Les skills sont exposés comme tools avec préfixe `skill_`.

### Mémoire 3 couches
Chaque interaction passe par : `store_interaction()` → Resource Layer (brut) → `_extract_items()` (heuristique) → Item Layer (unités) → `categories.organize()` (agrégation). La recherche hybride combine TF-IDF keyword, contexte conversationnel et embeddings vector (ChromaDB).

### Swarm Pipeline
`SwarmOrchestrator.execute()` : Planner → Coder (avec repo_map) → dry_run (py_compile) → Reviewer → routage conditionnel (ROUTE:security/tester) → Critic → validation ou boucle.

### Sandbox
Les commandes shell sont classifiées (DANGEROUS_PATTERNS vs SAFE_COMMANDS). Les dangereuses sont routées vers un conteneur Docker isolé. Le self-healing loop ré-envoie les erreurs au LLM pour correction automatique.

## Bugs connus / dette technique

### Priorité haute
1. **Streaming + tool calling incompatible** : `_stream_anthropic()` et `_stream_openai()` ne gèrent pas les `tool_use` blocks en mode streaming. L'agent ne peut pas exécuter d'outils quand le streaming est activé (mode par défaut de la WebUI).
2. **Extraction mémoire heuristique fragile** : `_analyze_content()` utilise du pattern-matching statique ("my name is", "je m'appelle"). Rate les reformulations et contextes implicites.
3. **Sous-agents séquentiels** : `_handle_delegation()` exécute les subtasks avec `await` une par une. Les tâches indépendantes devraient utiliser `asyncio.gather()`.
4. **Test coverage ~1.5%** : Seul `tests/test_gateway.py` (149 lignes) existe pour ~10K lignes de code.
5. **Auth désactivée par défaut** (SEC-004) : `api_key_required: false` dans `default.yaml`.

### Priorité moyenne
6. **Pas de channel adapters** : Pas de Telegram, WhatsApp, Discord. Terminal + Web uniquement.
7. **Analyse de délégation coûteuse** : `_analyze_task()` fait un appel LLM complet pour chaque message, même les questions simples.
8. **Pas de tool calling loop** : Après exécution d'un tool, le résultat n'est pas renvoyé au LLM pour continuation (single-shot).

## Tâches prioritaires (v1.1)

### T1 — Tests unitaires et d'intégration
Écrire des tests pytest pour les modules critiques :
- `tests/test_brain.py` : mock LLM, vérifier tool definitions, tool execution routing
- `tests/test_memory.py` : store/search/extract/forget cycle complet
- `tests/test_orchestrator.py` : délégation, analyse de tâche, agrégation
- `tests/test_sandbox.py` : classification dangerous/safe, path validation
- `tests/test_swarm.py` : pipeline Coder→Reviewer→Critic, dry_run
- `tests/test_mcp.py` : client connect/disconnect, tool invocation
- `tests/test_skills.py` : loader discovery, router matching, skill execution
- Objectif : >60% coverage. Utiliser `pytest-asyncio` pour les tests async, `unittest.mock.AsyncMock` pour les mocks.

### T2 — Tool calling loop (agentic loop)
Transformer `brain.generate()` en boucle agentique :
1. Appeler le LLM
2. Si `tool_calls` non vide → exécuter les tools → formater les résultats comme `tool_result` messages
3. Renvoyer au LLM avec les résultats
4. Répéter jusqu'à réponse texte finale ou max_iterations (défaut: 10)
5. Gérer aussi en mode streaming (`generate_stream()` avec accumulation des tool_use blocks)

### T3 — Streaming + tool use
Implémenter l'accumulation des tool_use blocks dans `_stream_anthropic()` :
- Accumuler les `content_block_start` (type=tool_use) et `content_block_delta` (input_json_delta)
- À `message_stop`, exécuter les tools et relancer le stream
- Émettre un événement SSE spécial (`type: tool_execution`) côté WebUI pour afficher l'état

### T4 — Adaptateur Telegram
Créer `openclaw/channels/telegram.py` :
- Utiliser `python-telegram-bot` (async)
- Adapter le pattern d'OpenClaw : normaliser les messages entrants → format interne → router vers brain
- Gérer les commandes `/start`, `/reset`, `/status`
- Support markdown dans les réponses
- Allowlist par user_id (comme le système approval existant)
- Config dans `default.yaml` sous `channels.telegram.token` et `channels.telegram.allowed_users`

### T5 — Auth par défaut + génération de clé API
- Changer `api_key_required: true` dans `default.yaml`
- Le wizard génère automatiquement une clé API (uuid4) au premier lancement
- Stocker dans `config/user.yaml`
- Middleware vérifie `X-API-Key` ou `?api_key=` sur toutes les routes sauf `/health`

### T6 — Parallélisation des sous-agents
Dans `orchestrator.py` `_handle_delegation()` :
- Remplacer la boucle séquentielle par `asyncio.gather(*[execute_subtask(s) for s in subtasks])`
- Ajouter un timeout par sous-agent (configurable)
- Gérer les erreurs partielles (certains sous-agents échouent, d'autres réussissent)

### T7 — Pré-filtre de délégation
Ajouter un classifieur léger avant `_analyze_task()` :
- Si message < 50 caractères ET ne contient pas de mots-clés complexes → skip délégation
- Mots-clés déclencheurs : "recherche", "compare", "analyse", "crée un", "écris un", "build", "implement"
- Économise un appel LLM complet sur ~80% des messages simples

## Fichiers à ne PAS modifier sans raison

- `config/default.yaml` : source de vérité pour les valeurs par défaut
- `memory/resource_layer.py` : les données brutes ne doivent JAMAIS être supprimées (contrat MemU)
- `agent/prompts/*.md` : templates de personnalité, modifiables par l'utilisateur

## Commandes utiles

```bash
# Lancer les tests
pytest tests/ -v --asyncio-mode=auto

# Lancer le projet
python run.py                    # Terminal + Gateway
python run.py terminal           # Terminal seul
python run.py gateway             # API seule

# Vérifier le code
python -m py_compile openclaw/agent/brain.py   # Vérif syntaxe rapide
grep -rn "TODO\|FIXME\|HACK" openclaw/         # Trouver la dette technique

# Compter les lignes
find openclaw -name "*.py" | xargs wc -l | tail -1
```

## Style de commit

```
[composant] description courte

Exemples :
[brain] implement agentic tool calling loop
[tests] add memory manager unit tests
[telegram] add channel adapter with allowlist
[security] enable API key auth by default
```
