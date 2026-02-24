"""
Tests for openclaw/tools/doctor.py — system diagnostics.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openclaw.tools.doctor import (
    CheckResult,
    DiagnosticReport,
    _check_channels,
    _check_config,
    _check_disk_space,
    _check_memory,
    _check_providers,
    _check_sandbox,
    _check_skills,
    run_diagnostics,
)


# ── Helpers ──────────────────────────────────────────────────


class FakeSettings:
    _base_dir = None
    _config_path = None

    def __init__(self, data=None, base_dir=None, config_path=None):
        self._data = data or {}
        self._base_dir = Path(base_dir) if base_dir else None
        self._config_path = Path(config_path) if config_path else None

    def get(self, dotpath, default=None):
        return self._data.get(dotpath, default)


# ══════════════════════════════════════════════════════════════
#  DiagnosticReport
# ══════════════════════════════════════════════════════════════


class TestDiagnosticReport:

    def test_empty_report_is_healthy(self):
        r = DiagnosticReport()
        assert r.healthy is True
        assert r.ok_count == 0
        assert r.warn_count == 0
        assert r.fail_count == 0

    def test_counts(self):
        r = DiagnosticReport(checks=[
            CheckResult("a", "OK", "ok"),
            CheckResult("b", "WARN", "warn"),
            CheckResult("c", "FAIL", "fail"),
            CheckResult("d", "OK", "ok"),
        ])
        assert r.ok_count == 2
        assert r.warn_count == 1
        assert r.fail_count == 1
        assert r.healthy is False

    def test_healthy_with_warnings(self):
        r = DiagnosticReport(checks=[
            CheckResult("a", "OK", "ok"),
            CheckResult("b", "WARN", "warn"),
        ])
        assert r.healthy is True

    def test_to_dict(self):
        r = DiagnosticReport(checks=[
            CheckResult("a", "OK", "all good"),
        ])
        d = r.to_dict()
        assert d["healthy"] is True
        assert d["ok"] == 1
        assert d["warn"] == 0
        assert d["fail"] == 0
        assert len(d["checks"]) == 1
        assert d["checks"][0]["name"] == "a"
        assert d["checks"][0]["status"] == "OK"

    def test_to_dict_includes_details(self):
        r = DiagnosticReport(checks=[
            CheckResult("x", "WARN", "msg", details="extra"),
        ])
        assert r.to_dict()["checks"][0]["details"] == "extra"


# ══════════════════════════════════════════════════════════════
#  _check_config
# ══════════════════════════════════════════════════════════════


class TestCheckConfig:

    def test_valid_config(self, tmp_path):
        import yaml
        cfg_file = tmp_path / "default.yaml"
        cfg_file.write_text(yaml.dump({
            "app": {}, "gateway": {}, "agent": {},
            "providers": {}, "memory": {}, "tools": {},
        }))
        settings = FakeSettings(config_path=str(cfg_file))
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_config()
        assert result.status == "OK"

    def test_missing_sections(self, tmp_path):
        import yaml
        cfg_file = tmp_path / "default.yaml"
        cfg_file.write_text(yaml.dump({"app": {}}))
        settings = FakeSettings(config_path=str(cfg_file))
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_config()
        assert result.status == "WARN"
        assert "Missing sections" in result.message

    def test_missing_file(self, tmp_path):
        settings = FakeSettings(config_path=str(tmp_path / "nope.yaml"))
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_config()
        assert result.status == "FAIL"

    def test_invalid_yaml(self, tmp_path):
        cfg_file = tmp_path / "default.yaml"
        cfg_file.write_text(": :\n  bad: [")
        settings = FakeSettings(config_path=str(cfg_file))
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_config()
        assert result.status == "FAIL"


# ══════════════════════════════════════════════════════════════
#  _check_providers
# ══════════════════════════════════════════════════════════════


class TestCheckProviders:

    def test_no_provider_enabled(self):
        settings = FakeSettings({
            "providers.anthropic.enabled": False,
            "providers.openai.enabled": False,
            "providers.ollama.enabled": False,
            "providers.custom.enabled": False,
        })
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_providers()
        assert result.status == "FAIL"

    def test_provider_enabled_with_key(self):
        settings = FakeSettings({
            "providers.anthropic.enabled": True,
            "providers.anthropic.api_key": "sk-test",
            "providers.openai.enabled": False,
            "providers.ollama.enabled": False,
            "providers.custom.enabled": False,
        })
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_providers()
        assert result.status == "OK"
        assert "anthropic" in result.message

    def test_provider_enabled_missing_key(self):
        settings = FakeSettings({
            "providers.anthropic.enabled": True,
            "providers.anthropic.api_key": "",
            "providers.openai.enabled": False,
            "providers.ollama.enabled": False,
            "providers.custom.enabled": False,
        })
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_providers()
        assert result.status == "WARN"
        assert "missing API key" in result.message

    def test_ollama_no_key_needed(self):
        settings = FakeSettings({
            "providers.anthropic.enabled": False,
            "providers.openai.enabled": False,
            "providers.ollama.enabled": True,
            "providers.ollama.api_key": "",
            "providers.custom.enabled": False,
        })
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_providers()
        assert result.status == "OK"


# ══════════════════════════════════════════════════════════════
#  _check_memory
# ══════════════════════════════════════════════════════════════


class TestCheckMemory:

    def test_memory_disabled(self):
        settings = FakeSettings({"memory.enabled": False})
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_memory()
        assert result.status == "WARN"
        assert "disabled" in result.message

    def test_memory_writable(self, tmp_path):
        store = tmp_path / "store"
        store.mkdir()
        settings = FakeSettings(
            {"memory.enabled": True, "memory.store_path": str(store)},
            base_dir=str(tmp_path),
        )
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_memory()
        assert result.status == "OK"
        assert "writable" in result.message

    def test_memory_creates_dir(self, tmp_path):
        store = tmp_path / "new_store"
        settings = FakeSettings(
            {"memory.enabled": True, "memory.store_path": str(store)},
            base_dir=str(tmp_path),
        )
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_memory()
        assert result.status == "OK"
        assert store.exists()


# ══════════════════════════════════════════════════════════════
#  _check_disk_space
# ══════════════════════════════════════════════════════════════


class TestCheckDiskSpace:

    def test_ok(self):
        result = _check_disk_space()
        # On any real system we should get OK or WARN but not an exception
        assert result.status in ("OK", "WARN", "FAIL")
        assert result.name == "disk"

    def test_low_disk_returns_warn(self):
        """Simulate low disk via mock."""
        import shutil
        fake_usage = shutil.disk_usage.__class__
        with patch("openclaw.tools.doctor.shutil.disk_usage") as mock_du:
            mock_du.return_value = type("Usage", (), {
                "total": 100_000_000_000,
                "free": 10_000_000_000,  # 10%
                "used": 90_000_000_000,
            })()
            result = _check_disk_space()
        assert result.status == "WARN"

    def test_critical_disk_returns_fail(self):
        with patch("openclaw.tools.doctor.shutil.disk_usage") as mock_du:
            mock_du.return_value = type("Usage", (), {
                "total": 100_000_000_000,
                "free": 2_000_000_000,  # 2%
                "used": 98_000_000_000,
            })()
            result = _check_disk_space()
        assert result.status == "FAIL"


# ══════════════════════════════════════════════════════════════
#  _check_skills
# ══════════════════════════════════════════════════════════════


class TestCheckSkills:

    def test_skills_found(self, tmp_path):
        builtin = tmp_path / "skills" / "builtin"
        (builtin / "web_search").mkdir(parents=True)
        (builtin / "web_search" / "SKILL.md").write_text("# Web Search")
        settings = FakeSettings(
            {"skills.builtin_path": "skills/builtin"},
            base_dir=str(tmp_path),
        )
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_skills()
        assert result.status == "OK"
        assert "1 skill" in result.message
        assert "web_search" in result.message

    def test_no_skills(self, tmp_path):
        builtin = tmp_path / "skills" / "builtin"
        builtin.mkdir(parents=True)
        settings = FakeSettings(
            {"skills.builtin_path": "skills/builtin"},
            base_dir=str(tmp_path),
        )
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_skills()
        assert result.status == "WARN"

    def test_no_base_dir(self):
        settings = FakeSettings({"skills.builtin_path": "skills/builtin"})
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_skills()
        assert result.status == "WARN"


# ══════════════════════════════════════════════════════════════
#  _check_channels
# ══════════════════════════════════════════════════════════════


class TestCheckChannels:

    def test_no_channels(self):
        settings = FakeSettings({"channels.telegram.enabled": False})
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_channels()
        assert result.status == "OK"
        assert "terminal only" in result.message

    def test_telegram_enabled_with_token(self):
        settings = FakeSettings({
            "channels.telegram.enabled": True,
            "channels.telegram.token": "123:ABC",
        })
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_channels()
        assert result.status == "OK"
        assert "telegram" in result.message

    def test_telegram_enabled_no_token(self):
        settings = FakeSettings({
            "channels.telegram.enabled": True,
            "channels.telegram.token": "",
        })
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_channels()
        assert result.status == "WARN"


# ══════════════════════════════════════════════════════════════
#  _check_sandbox
# ══════════════════════════════════════════════════════════════


class TestCheckSandbox:

    def test_sandbox_disabled(self):
        settings = FakeSettings({"sandbox.enabled": False})
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            result = _check_sandbox()
        assert result.status == "WARN"
        assert "disabled" in result.message

    def test_docker_reachable(self):
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_docker = MagicMock()
        mock_docker.from_env.return_value = mock_client
        settings = FakeSettings({"sandbox.enabled": True})
        with patch("openclaw.tools.doctor.get_settings", return_value=settings), \
             patch.dict("sys.modules", {"docker": mock_docker}):
            result = _check_sandbox()
        assert result.status == "OK"

    def test_docker_not_installed(self):
        settings = FakeSettings({"sandbox.enabled": True})
        with patch("openclaw.tools.doctor.get_settings", return_value=settings), \
             patch.dict("sys.modules", {"docker": None}):
            result = _check_sandbox()
        assert result.status == "WARN"


# ══════════════════════════════════════════════════════════════
#  run_diagnostics
# ══════════════════════════════════════════════════════════════


class TestRunDiagnostics:

    def test_returns_report_with_all_checks(self, tmp_path):
        import yaml
        cfg_file = tmp_path / "default.yaml"
        cfg_file.write_text(yaml.dump({
            "app": {}, "gateway": {}, "agent": {},
            "providers": {}, "memory": {}, "tools": {},
        }))
        store = tmp_path / "store"
        store.mkdir()
        settings = FakeSettings(
            data={
                "memory.enabled": True,
                "memory.store_path": str(store),
                "providers.anthropic.enabled": True,
                "providers.anthropic.api_key": "sk-x",
                "providers.openai.enabled": False,
                "providers.ollama.enabled": False,
                "providers.custom.enabled": False,
                "skills.builtin_path": "skills/builtin",
                "channels.telegram.enabled": False,
                "sandbox.enabled": False,
            },
            base_dir=str(tmp_path),
            config_path=str(cfg_file),
        )
        with patch("openclaw.tools.doctor.get_settings", return_value=settings):
            report = run_diagnostics()

        assert len(report.checks) == 7
        names = [c.name for c in report.checks]
        assert "config" in names
        assert "providers" in names
        assert "memory" in names
        assert "disk" in names
        assert "skills" in names
        assert "channels" in names
        assert "sandbox" in names
