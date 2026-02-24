"""
/doctor diagnostics – checks system health across all subsystems.

Each check returns a :class:`CheckResult` with status OK / WARN / FAIL.
``run_diagnostics()`` aggregates all checks into a single report.
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.tools.doctor")


# ── Data types ──────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    status: str  # "OK", "WARN", "FAIL"
    message: str
    details: str = ""


@dataclass
class DiagnosticReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "OK")

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "WARN")

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "FAIL")

    @property
    def healthy(self) -> bool:
        return self.fail_count == 0

    def to_dict(self) -> dict:
        return {
            "healthy": self.healthy,
            "ok": self.ok_count,
            "warn": self.warn_count,
            "fail": self.fail_count,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


# ── Individual checks ───────────────────────────────────────

def _check_config() -> CheckResult:
    """Verify the YAML config is parseable and contains required keys."""
    settings = get_settings()
    try:
        cfg_path = settings._config_path
        if cfg_path and cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return CheckResult("config", "FAIL", "default.yaml is not a mapping")
        else:
            return CheckResult("config", "FAIL", "default.yaml not found")

        # Check required top-level sections
        required = {"app", "gateway", "agent", "providers", "memory", "tools"}
        missing = required - set(data.keys())
        if missing:
            return CheckResult(
                "config", "WARN",
                f"Missing sections: {', '.join(sorted(missing))}",
            )

        return CheckResult("config", "OK", "Configuration valid")
    except yaml.YAMLError as e:
        return CheckResult("config", "FAIL", f"YAML parse error: {e}")
    except Exception as e:
        return CheckResult("config", "FAIL", str(e))


def _check_providers() -> CheckResult:
    """Check that at least one LLM provider is enabled and has an API key."""
    settings = get_settings()
    enabled = []
    missing_key = []

    for name in ("anthropic", "openai", "ollama", "custom"):
        if settings.get(f"providers.{name}.enabled", False):
            enabled.append(name)
            key = settings.get(f"providers.{name}.api_key", "")
            if name not in ("ollama",) and not key:
                missing_key.append(name)

    if not enabled:
        return CheckResult("providers", "FAIL", "No LLM provider enabled")

    if missing_key:
        return CheckResult(
            "providers", "WARN",
            f"Enabled but missing API key: {', '.join(missing_key)}",
            details=f"Enabled: {', '.join(enabled)}",
        )

    return CheckResult(
        "providers", "OK",
        f"Providers active: {', '.join(enabled)}",
    )


def _check_memory() -> CheckResult:
    """Check that the memory store path is accessible."""
    settings = get_settings()
    if not settings.get("memory.enabled", True):
        return CheckResult("memory", "WARN", "Memory system disabled")

    store_path = settings.get("memory.store_path", "memory/store")
    base_dir = settings._base_dir
    if base_dir:
        resolved = base_dir / store_path if not Path(store_path).is_absolute() else Path(store_path)
    else:
        resolved = Path(store_path)

    if resolved.exists():
        # Quick write test
        try:
            probe = resolved / ".doctor_probe"
            probe.write_text("ok")
            probe.unlink()
            return CheckResult("memory", "OK", f"Memory store writable: {resolved}")
        except OSError as e:
            return CheckResult("memory", "FAIL", f"Memory store not writable: {e}")
    else:
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            return CheckResult("memory", "OK", f"Memory store created: {resolved}")
        except OSError as e:
            return CheckResult("memory", "FAIL", f"Cannot create memory store: {e}")


def _check_disk_space() -> CheckResult:
    """Check available disk space on the filesystem root."""
    try:
        usage = shutil.disk_usage("/")
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        pct_free = (usage.free / usage.total) * 100

        if pct_free < 5:
            return CheckResult(
                "disk", "FAIL",
                f"Critically low disk space: {free_gb:.1f} GB free ({pct_free:.0f}%)",
            )
        if pct_free < 15:
            return CheckResult(
                "disk", "WARN",
                f"Low disk space: {free_gb:.1f} GB free ({pct_free:.0f}%)",
            )
        return CheckResult(
            "disk", "OK",
            f"{free_gb:.1f} GB free / {total_gb:.1f} GB total ({pct_free:.0f}% free)",
        )
    except Exception as e:
        return CheckResult("disk", "WARN", f"Cannot check disk space: {e}")


def _check_skills() -> CheckResult:
    """Check that skills directory exists and contains skills."""
    settings = get_settings()
    base_dir = settings._base_dir
    if not base_dir:
        return CheckResult("skills", "WARN", "Base directory unknown")

    builtin_path = base_dir / settings.get("skills.builtin_path", "skills/builtin")
    if not builtin_path.exists():
        return CheckResult("skills", "WARN", "Skills directory not found")

    skill_dirs = [
        d for d in builtin_path.iterdir()
        if d.is_dir() and (d / "SKILL.md").exists()
    ]

    if not skill_dirs:
        return CheckResult("skills", "WARN", "No skills found")

    return CheckResult(
        "skills", "OK",
        f"{len(skill_dirs)} skill(s) loaded: {', '.join(d.name for d in skill_dirs)}",
    )


def _check_channels() -> CheckResult:
    """Check channel configuration."""
    settings = get_settings()
    active = []

    if settings.get("channels.telegram.enabled", False):
        token = settings.get("channels.telegram.token", "")
        if token:
            active.append("telegram")
        else:
            return CheckResult(
                "channels", "WARN",
                "Telegram enabled but no token configured",
            )

    if not active:
        return CheckResult("channels", "OK", "No external channels enabled (terminal only)")

    return CheckResult(
        "channels", "OK",
        f"Active channels: {', '.join(active)}",
    )


def _check_sandbox() -> CheckResult:
    """Check Docker availability for sandbox execution."""
    settings = get_settings()
    if not settings.get("sandbox.enabled", True):
        return CheckResult("sandbox", "WARN", "Sandbox disabled in config")

    try:
        import docker
        client = docker.from_env()
        client.ping()
        return CheckResult("sandbox", "OK", "Docker daemon reachable")
    except ImportError:
        return CheckResult("sandbox", "WARN", "docker package not installed")
    except Exception as e:
        err = str(e)
        if "connection" in err.lower() or "permission" in err.lower():
            return CheckResult("sandbox", "FAIL", f"Docker unreachable: {err[:120]}")
        return CheckResult("sandbox", "WARN", f"Docker check failed: {err[:120]}")


# ── Main entry point ────────────────────────────────────────

def run_diagnostics() -> DiagnosticReport:
    """Run all diagnostic checks and return a report."""
    report = DiagnosticReport()
    report.checks.append(_check_config())
    report.checks.append(_check_providers())
    report.checks.append(_check_memory())
    report.checks.append(_check_disk_space())
    report.checks.append(_check_skills())
    report.checks.append(_check_channels())
    report.checks.append(_check_sandbox())
    return report
