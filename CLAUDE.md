# CLAUDE.md — NexusMind Project Context

This file provides project context for AI assistants (Claude Code, Copilot, etc.)
working on the NexusMind / OpenClaw codebase.

## README

- **English**: [README.md](README.md)
- **French**: [README.fr.md](README.fr.md)

## Project Overview

**OpenClaw NexusMind v1.0.0** — Autonomous AI Agent Framework (Python 3.11+, MIT).

Combines three architectures:
- **OpenClaw**: Gateway-centric API, modular skills, system execution
- **MemU**: 3-layer persistent memory with graceful forgetting
- **AgentZero**: Hierarchical delegation, tool creation, prompt-driven behavior

## Test Suite

- **654 tests** across 19 test files, all passing
- Framework: pytest + pytest-asyncio
- Run: `pytest tests/ -v`
- Config: `pyproject.toml` → `[tool.pytest.ini_options]`

### Test Files

| File                     | Module Tested                | Tests |
|--------------------------|------------------------------|------:|
| test_memory.py           | memory/manager.py            |   ~30 |
| test_sqlite_memory.py    | memory/ (SQLite FTS5)        |   ~25 |
| test_brain.py            | agent/brain.py               |   ~40 |
| test_orchestrator.py     | agent/orchestrator.py        |   ~20 |
| test_swarm.py            | agent/swarm.py               |   ~35 |
| test_scheduler.py        | agent/scheduler.py           |    25 |
| test_context.py          | agent/context.py             |    21 |
| test_sandbox.py          | sandbox/                     |   ~20 |
| test_gateway.py          | gateway/server.py            |   ~30 |
| test_api_auth.py         | gateway/ (auth)              |   ~15 |
| test_mcp.py              | mcp/                         |   ~40 |
| test_skills.py           | skills/                      |   ~39 |
| test_doctor.py           | tools/doctor.py              |   ~15 |
| test_telegram.py         | channels/telegram.py         |   ~30 |
| test_discord.py          | channels/discord.py          |    30 |
| test_providers_new.py    | providers/ (gemini, openrouter, ollama) | 23 |
| test_tracing.py          | tracing/recorder.py          |    38 |
| test_workspace.py        | tools/ (workspace scoping)   |   ~15 |
| test_secrets.py          | security/ (Fernet encryption)|   ~10 |

## Providers (6)

| Provider   | Module                              | Notes              |
|------------|-------------------------------------|--------------------|
| Anthropic  | providers/anthropic_provider.py     | anthropic SDK      |
| OpenAI     | providers/openai_provider.py        | openai SDK         |
| Gemini     | providers/gemini_provider.py        | httpx REST (no SDK)|
| OpenRouter | providers/openrouter_provider.py    | Inherits OpenAI    |
| Ollama     | (via OpenAI provider)               | Auto tool detection|
| Custom     | (via OpenAI provider)               | User-configurable  |

Factory: `providers/factory.py` → `create_provider(name, settings)`

## Channels (4)

| Channel   | Module                   | Notes                        |
|-----------|--------------------------|------------------------------|
| Terminal  | ui/terminal.py           | Rich-based, local only       |
| Web UI    | ui/web/app.py            | FastAPI + Jinja2             |
| Telegram  | channels/telegram.py     | python-telegram-bot, allowlist |
| Discord   | channels/discord.py      | discord.py, deny-by-default  |

## Key Modules

```
openclaw/
├── main.py                  # Entry point, CLI (argparse), doctor mode
├── config/
│   ├── default.yaml         # Default configuration (all settings)
│   └── settings.py          # Settings manager with env override
├── agent/
│   ├── brain.py             # Core agentic loop, multi-provider
│   ├── orchestrator.py      # Multi-agent orchestration
│   ├── swarm.py             # 7-role swarm pipeline
│   ├── scheduler.py         # Agentic cron (async task scheduling)
│   └── context.py           # Token counting, context compression
├── providers/
│   ├── base.py              # ProviderBase ABC
│   ├── factory.py           # Provider factory + Ollama tool detect
│   ├── anthropic_provider.py
│   ├── openai_provider.py
│   ├── gemini_provider.py
│   └── openrouter_provider.py
├── memory/
│   ├── manager.py           # 3-layer memory manager
│   ├── resource_layer.py    # Layer 1: raw data (never deleted)
│   ├── item_layer.py        # Layer 2: extracted memory units
│   ├── category_layer.py    # Layer 3: aggregated documents
│   └── retrieval.py         # Hybrid search (keyword + semantic)
├── channels/
│   ├── base.py              # ChannelBase ABC
│   ├── telegram.py          # Telegram bot adapter
│   └── discord.py           # Discord bot adapter
├── tracing/
│   └── recorder.py          # Span/Trace/TraceRecorder, persistence
├── security/
│   └── secrets.py           # Fernet encryption for user.yaml secrets
├── mcp/
│   ├── client.py            # MCP client (stdio/SSE)
│   └── registry.py          # Multi-server registry + approval
├── sandbox/
│   ├── container.py         # Docker container lifecycle
│   └── executor.py          # Self-healing code execution
├── skills/
│   ├── loader.py            # Auto-discovery, dynamic loading
│   └── router.py            # Intent routing
├── tools/
│   ├── executor.py          # 4 base tools: shell, read, write, search
│   ├── doctor.py            # System diagnostics (7 checks)
│   └── repo_map.py          # AST-based repository map
├── gateway/
│   ├── server.py            # FastAPI + WS + SSE
│   ├── middleware.py         # Rate limiting, security
│   └── approval.py          # Human-in-the-loop MCP approval
└── ui/
    ├── terminal.py          # Rich terminal UI
    └── web/                 # Web UI (SPA, dark theme)
```

## Conventions

- **Commit format**: `[type] message` — feat, fix, refactor, tests, docs, chore, security
- **Dependencies**: `pyproject.toml` with optional extras (ml, telegram, discord, providers, docker, monitoring, dev, all)
- **Config**: YAML-based with env var overrides (`OPENCLAW_SECTION__KEY`)
- **Lint**: `ruff check openclaw/` (line-length 100, Python 3.11)
- **Access control**: Deny-by-default for channels (allowlists), refuse public bind for gateway
