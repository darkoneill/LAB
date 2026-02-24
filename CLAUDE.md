# CLAUDE.md — NexusMind v1.0.0 (OpenClaw Python Clone)

## Projet

NexusMind est un assistant IA autonome en Python (~12K lignes) fusionnant les architectures d'OpenClaw (gateway), MemU (mémoire 3 couches) et AgentZero (délégation hiérarchique). Le repo est dans `openclaw/`.

## Stack technique

- **Langage** : Python 3.11+, 100% async (asyncio)
- **Framework web** : FastAPI + Uvicorn + WebSocket + SSE
- **LLM** : Anthropic SDK (natif) + OpenAI-compatible (Ollama, etc.)
- **Mémoire** : ChromaDB (vector, optionnel), SQLite FTS5 (défaut léger), scikit-learn TF-IDF
- **Sandbox** : Docker SDK (isolation des commandes dangereuses)
- **UI** : Rich (terminal) + Jinja2/vanilla JS (web)
- **Tests** : pytest + pytest-asyncio (~2200 lignes, 4 fichiers)
- **Config** : YAML (pydantic-settings), variables d'env `OPENCLAW_*`

## Architecture clé

```
openclaw/
├── providers/              # ABC ProviderBase + implémentations (à créer)
├── channels/               # ABC ChannelBase + adaptateurs (à créer)
├── agent/
│   ├── brain.py            # Moteur LLM + agentic tool calling loop
│   ├── orchestrator.py     # Délégation hiérarchique
│   ├── swarm.py            # Multi-agent spécialisé (Coder→Reviewer→Critic)
│   ├── scheduler.py        # Cron agentic
│   └── context.py          # Compression de contexte
├── memory/
│   ├── manager.py          # Gestionnaire autonome 3 couches
│   ├── backends/           # ABC MemoryBackend + sqlite/chromadb (à créer)
│   ├── resource_layer.py   # Couche 1 : données brutes (JAMAIS supprimées)
│   ├── item_layer.py       # Couche 2 : unités extraites + scoring
│   ├── category_layer.py   # Couche 3 : documents agrégés + évolution
│   ├── retrieval.py        # Recherche hybride
│   ├── vector_store.py     # ChromaDB wrapper (lazy-loaded)
│   └── evolution.py        # Réflexion périodique
├── gateway/
│   ├── server.py           # FastAPI : REST + WS + SSE + sessions + APIKeyMiddleware
│   ├── router.py           # Routage multi-provider avec failover
│   ├── middleware.py        # Rate limiting, sécurité, cache sémantique
│   └── approval.py         # Système d'approbation
├── sandbox/
│   ├── executor.py         # Auto-routing sandbox + self-healing code loop
│   └── container.py        # Gestion conteneurs Docker isolés
├── mcp/                    # Client/Server MCP (stdio + SSE)
├── skills/                 # Système modulaire (BaseSkill ABC + loader + router)
├── tools/
│   ├── executor.py         # shell, read_file, write_file, search_files
│   ├── repo_map.py         # Squelette AST pour le Swarm
│   └── doctor.py           # Diagnostics système (à créer)
├── tracing/recorder.py     # Observabilité : spans, latence, coûts
├── ui/                     # Terminal Rich + Web SPA
├── config/
│   ├── settings.py         # Singleton Settings
│   └── default.yaml        # Configuration par défaut
├── main.py                 # Point d'entrée
└── setup_wizard.py         # Installation guidée (génère API key)
```

## Conventions de code

- Tout est **async/await**. Pas de code bloquant dans la boucle événementielle.
- Imports **relatifs** dans le package (`from .resource_layer import ResourceLayer`).
- Logger par module : `logger = logging.getLogger("openclaw.module.submodule")`
- Settings via `get_settings().get("section.key", default)`.
- **Dataclasses** ou **pydantic BaseModel** pour les structures.
- Type hints obligatoires sur les signatures publiques.
- Docstrings en anglais, commentaires en français acceptés.

## Patterns importants

### Agentic Tool Calling Loop (T2 — IMPLÉMENTÉ)
`brain.generate()` boucle: appel LLM → si tool_calls → exécute via `_execute_tools()` → formate en tool_result messages → relance LLM. Max 10 rounds. Les messages tool_calls et tool_results sont formatés différemment pour Anthropic (content blocks) et OpenAI (tool role messages).

### API Key Auth (T5 — IMPLÉMENTÉ)
`APIKeyMiddleware` protège toutes les routes `/api/*`. `api_key_required: true` par défaut. Le wizard génère un UUID au premier lancement, stocké dans `user.yaml`. Routes exemptées : `/health`, `/ws/`, statiques, OPTIONS.

### Mémoire 3 couches
`store_interaction()` → Resource Layer (brut) → `_extract_items()` (heuristique) → Item Layer → `categories.organize()`. Recherche hybride TF-IDF + contextuel + embeddings vector.

### Swarm Pipeline
`SwarmOrchestrator.execute()` : Planner → Coder (avec repo_map) → dry_run → Reviewer → routage ROUTE:security/tester → Critic → validation ou boucle.

### Sandbox
Commandes classifiées DANGEROUS_PATTERNS vs SAFE_COMMANDS. Dangereuses → conteneur Docker isolé. Self-healing loop renvoie erreurs au LLM.

## Principes de sécurité (inspirés ZeroClaw)

- **Deny-by-default** : allowlists vides = aucun accès. Pas de `"*"` implicite.
- **Workspace scoping** : les file tools doivent résoudre les paths canoniques et vérifier qu'ils restent dans le workspace. Détection symlink escape.
- **Gateway localhost-only** : bind `127.0.0.1` par défaut. Refuser `0.0.0.0` sans config explicite.
- **Fail loud** : jamais de fallback silencieux. Erreur explicite si config invalide.
- **Pas de secrets dans les logs** : redaction récursive des clés API, tokens, passwords.

## Bugs connus / dette technique

### Priorité haute
1. **Streaming + tool calling incompatible** : `generate_stream()` ne gère pas les tool_use blocks
2. **Pas d'interfaces ABC formelles** : couplage implicite entre brain et providers
3. **ChromaDB obligatoire pour la mémoire vectorielle** : ~500MB RAM, pas d'alternative légère
4. **Tests manquants** : orchestrator, swarm, sandbox, mcp, skills à 0% coverage
5. **Pas de channel adapters** : Terminal + Web uniquement

### Priorité moyenne
6. **Workspace scoping absent** sur les file tools (path traversal possible hors blocked_paths)
7. **Sous-agents séquentiels** dans orchestrator
8. **Pas de commande doctor** pour diagnostiquer les problèmes
9. **Skills chargées sans audit de sécurité**

## Tâches (ordre d'exécution recommandé)

### P1 — Interfaces ABC + Factory Pattern
Créer ProviderBase, ChannelBase, MemoryBackend comme ABC. Extraire la logique d'appel LLM de brain.py dans des classes provider concrètes. Factory pattern depuis la config.

### P2 — Backend mémoire SQLite FTS5
Alternative légère à ChromaDB : SQLite + FTS5 BM25. Réduire RAM de ~500MB à ~50MB. Garder ChromaDB en option.

### P3 — Tests orchestrator + swarm + sandbox
Modules critiques à 0% coverage. Mock brain.generate(), tester pipelines complets.

### P4 — Streaming + tool calling
Accumulation tool_use blocks en streaming Anthropic. Nouveau `generate_stream_with_tools()`.

### P5 — Adaptateur Telegram
Premier channel. Deny-by-default allowlist (pattern ZeroClaw). Commandes /start, /reset, /status.

### P6 — Doctor + workspace scoping
Commande diagnostic. Workspace scoping strict sur les file tools avec détection symlink.

## Fichiers à ne PAS modifier sans raison

- `config/default.yaml` : source de vérité pour les valeurs par défaut
- `memory/resource_layer.py` : données brutes JAMAIS supprimées (contrat MemU)
- `agent/prompts/*.md` : templates de personnalité, modifiables par l'utilisateur

## Commandes utiles

```bash
pytest tests/ -v --asyncio-mode=auto          # Tous les tests
python run.py                                  # Terminal + Gateway
python run.py gateway                          # API seule
grep -rn "TODO\|FIXME\|HACK" openclaw/        # Dette technique
find openclaw -name "*.py" | xargs wc -l | tail -1  # Compter les lignes
```

## Style de commit

```
[composant] description courte

Exemples :
[providers] extract ABC ProviderBase + factory pattern
[memory] add SQLite FTS5 backend as lightweight default
[tests] add orchestrator + swarm test suites
[channels] add Telegram adapter with deny-by-default allowlist
[security] add workspace scoping with symlink escape detection
```
