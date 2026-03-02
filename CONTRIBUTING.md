# Contributing to NexusMind

Thanks for your interest in contributing! This document covers the essentials.

## Workflow

1. **Fork** the repo and clone your fork.
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
3. **Make your changes**, keeping commits focused and atomic.
4. **Run tests and lint** before submitting:
   ```bash
   pip install -e ".[dev]"
   pytest tests/ -v
   ruff check openclaw/
   ```
5. **Push** and open a **Pull Request** against `main`.

## Commit Convention

Use the format `[type] short description` where type is one of:

| Type       | Usage                                |
|------------|--------------------------------------|
| `feat`     | New feature                          |
| `fix`      | Bug fix                              |
| `refactor` | Code restructuring (no behavior change) |
| `tests`    | Adding or updating tests             |
| `docs`     | Documentation only                   |
| `chore`    | Build, CI, tooling, dependencies     |
| `security` | Security fix or hardening            |

Examples:
```
[feat] add Gemini provider with streaming support
[fix] prevent null byte path traversal in workspace
[tests] add scheduler, context, tracing test suites
```

## Pull Request Guidelines

- PRs must pass CI (tests + lint) before merge.
- Keep PRs small and focused — one feature or fix per PR.
- Include a short description of **what** and **why** in the PR body.
- Add tests for new features and bug fixes.

## Project Structure

- `openclaw/` — main package (providers, channels, agent, memory, etc.)
- `tests/` — pytest test suite (654+ tests)
- `openclaw/config/default.yaml` — default configuration reference

## Questions?

Open an issue on GitHub or check the existing documentation in the repo.
