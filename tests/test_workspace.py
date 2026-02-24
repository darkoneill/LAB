"""
Tests for workspace scoping in openclaw/tools/executor.py.
Covers: null-byte rejection, symlink escape detection,
workspace boundary enforcement, and config-driven toggling.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openclaw.tools.executor import ToolExecutor


# ── Helpers ──────────────────────────────────────────────────


class FakeSettings:
    _base_dir = None

    def __init__(self, overrides=None):
        self._data = {
            "tools.workspace_path": "",
            "tools.workspace_only": True,
            "tools.shell.enabled": True,
            "tools.shell.timeout_seconds": 10,
            "tools.shell.blocked_commands": [],
            "tools.file_manager.blocked_paths": [],
            "tools.file_manager.max_file_size_mb": 100,
        }
        if overrides:
            self._data.update(overrides)

    def get(self, dotpath, default=None):
        return self._data.get(dotpath, default)


def _make_executor(workspace_path="", workspace_only=True, blocked_paths=None):
    """Create a ToolExecutor with controlled workspace settings."""
    overrides = {
        "tools.workspace_path": workspace_path,
        "tools.workspace_only": workspace_only,
    }
    if blocked_paths:
        overrides["tools.file_manager.blocked_paths"] = blocked_paths
    settings = FakeSettings(overrides)
    with patch("openclaw.tools.executor.get_settings", return_value=settings):
        return ToolExecutor()


# ══════════════════════════════════════════════════════════════
#  Null-byte injection
# ══════════════════════════════════════════════════════════════


class TestNullByteBlocking:

    def test_null_byte_in_path(self):
        ex = _make_executor(workspace_only=False)
        ok, err = ex._validate_path("/home/user/file\x00.txt")
        assert ok is False
        assert "null byte" in err

    def test_null_byte_at_start(self):
        ex = _make_executor(workspace_only=False)
        ok, err = ex._validate_path("\x00/etc/passwd")
        assert ok is False
        assert "null byte" in err

    def test_null_byte_embedded(self):
        ex = _make_executor(workspace_only=False)
        ok, err = ex._validate_path("/tmp/safe\x00/../../../etc/shadow")
        assert ok is False
        assert "null byte" in err


# ══════════════════════════════════════════════════════════════
#  Workspace scoping
# ══════════════════════════════════════════════════════════════


class TestWorkspaceScoping:

    def test_path_inside_workspace_allowed(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        target = workspace / "src" / "main.py"
        target.parent.mkdir(parents=True)
        target.write_text("pass")

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        ok, err = ex._validate_path(str(target))
        assert ok is True
        assert err == ""

    def test_path_outside_workspace_denied(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        outside = tmp_path / "other" / "secret.txt"
        outside.parent.mkdir(parents=True)
        outside.write_text("secret")

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        ok, err = ex._validate_path(str(outside))
        assert ok is False
        assert "outside workspace" in err.lower()

    def test_workspace_only_disabled(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        outside = tmp_path / "other" / "file.txt"
        outside.parent.mkdir(parents=True)
        outside.write_text("data")

        ex = _make_executor(workspace_path=str(workspace), workspace_only=False)
        ok, err = ex._validate_path(str(outside))
        assert ok is True

    def test_dot_dot_traversal_blocked(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        # Try to escape via ../
        evil_path = str(workspace / ".." / "other" / "secret.txt")

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        ok, err = ex._validate_path(evil_path)
        assert ok is False
        assert "outside workspace" in err.lower()

    def test_workspace_root_itself_allowed(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        ok, err = ex._validate_path(str(workspace))
        assert ok is True

    def test_default_workspace_is_cwd(self):
        """When workspace_path is empty, defaults to cwd."""
        ex = _make_executor(workspace_path="", workspace_only=True)
        assert ex._workspace == Path.cwd().resolve()


# ══════════════════════════════════════════════════════════════
#  Symlink escape detection
# ══════════════════════════════════════════════════════════════


class TestSymlinkEscapeDetection:

    def test_symlink_inside_workspace_allowed(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        real_file = workspace / "real.txt"
        real_file.write_text("content")
        link = workspace / "link.txt"
        link.symlink_to(real_file)

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        ok, err = ex._validate_path(str(link))
        assert ok is True

    def test_symlink_escaping_workspace_denied(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        outside_file = tmp_path / "secret.txt"
        outside_file.write_text("secret")
        link = workspace / "escape.txt"
        link.symlink_to(outside_file)

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        ok, err = ex._validate_path(str(link))
        assert ok is False
        assert "outside workspace" in err.lower()

    def test_symlink_dir_escape_denied(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        outside_dir = tmp_path / "secrets"
        outside_dir.mkdir()
        (outside_dir / "key.pem").write_text("private")
        link = workspace / "data"
        link.symlink_to(outside_dir)

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        ok, err = ex._validate_path(str(link / "key.pem"))
        assert ok is False
        assert "outside workspace" in err.lower()


# ══════════════════════════════════════════════════════════════
#  Blocked paths still work with workspace scoping
# ══════════════════════════════════════════════════════════════


class TestBlockedPathsWithWorkspace:

    def test_blocked_path_inside_workspace(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        blocked = workspace / "secrets" / "creds.json"
        blocked.parent.mkdir(parents=True)
        blocked.write_text("{}")

        ex = _make_executor(
            workspace_path=str(workspace),
            workspace_only=True,
            blocked_paths=[str(blocked.parent)],
        )
        ok, err = ex._validate_path(str(blocked))
        assert ok is False
        assert "blocked" in err.lower()


# ══════════════════════════════════════════════════════════════
#  Integration: read_file / write_file / search_files
# ══════════════════════════════════════════════════════════════


class TestFileToolsWorkspaceIntegration:

    @pytest.mark.asyncio
    async def test_read_file_outside_workspace_denied(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("secret")

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        result = await ex._tool_read_file(path=str(outside))
        assert result["success"] is False
        assert "outside workspace" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_write_file_outside_workspace_denied(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        outside = tmp_path / "evil.txt"

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        result = await ex._tool_write_file(path=str(outside), content="pwned")
        assert result["success"] is False
        assert "outside workspace" in result["error"].lower()
        assert not outside.exists()

    @pytest.mark.asyncio
    async def test_search_files_outside_workspace_denied(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        outside = tmp_path / "other"
        outside.mkdir()

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        result = await ex._tool_search_files(path=str(outside), pattern="*")
        assert result["success"] is False
        assert "outside workspace" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_read_file_inside_workspace_ok(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        target = workspace / "hello.txt"
        target.write_text("hello world")

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        result = await ex._tool_read_file(path=str(target))
        assert result["success"] is True
        assert result["content"] == "hello world"

    @pytest.mark.asyncio
    async def test_write_file_inside_workspace_ok(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()
        target = workspace / "output.txt"

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        result = await ex._tool_write_file(path=str(target), content="data")
        assert result["success"] is True
        assert target.read_text() == "data"

    @pytest.mark.asyncio
    async def test_null_byte_in_read_file(self, tmp_path):
        workspace = tmp_path / "project"
        workspace.mkdir()

        ex = _make_executor(workspace_path=str(workspace), workspace_only=True)
        result = await ex._tool_read_file(path=str(workspace / "file\x00.txt"))
        assert result["success"] is False
        assert "null byte" in result["error"]
