"""
Tests for openclaw/skills/ — base.py, loader.py, router.py, builtin skills.
All I/O and subprocesses are mocked.
"""

import asyncio
import os
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw.skills.base import BaseSkill
from openclaw.skills.loader import SkillLoader
from openclaw.skills.router import SkillRouter


# ── BaseSkill ABC ────────────────────────────────────────────


class TestBaseSkillABC:
    def test_subclass_must_implement_execute(self):
        """Cannot instantiate a BaseSkill subclass without execute()."""

        class BadSkill(BaseSkill):
            name = "bad"
            # execute not implemented

        with pytest.raises(TypeError, match="execute"):
            BadSkill()

    def test_subclass_with_execute_ok(self):
        class GoodSkill(BaseSkill):
            name = "good"

            async def execute(self, **kwargs):
                return {"success": True}

        skill = GoodSkill()
        assert skill.name == "good"

    def test_get_info(self):
        class InfoSkill(BaseSkill):
            name = "info_test"
            description = "A test"
            version = "2.0"
            tags = ["test"]

            async def execute(self, **kwargs):
                return {}

        skill = InfoSkill()
        info = skill.get_info()
        assert info["name"] == "info_test"
        assert info["description"] == "A test"
        assert info["version"] == "2.0"
        assert info["tags"] == ["test"]
        assert info["enabled"] is True


class TestBaseSkillMatching:
    def _make_skill(self, name="test", description="A test skill", tags=None):
        class DummySkill(BaseSkill):
            async def execute(self, **kw):
                return {}

        skill = DummySkill.__new__(DummySkill)
        skill.name = name
        skill.description = description
        skill.tags = tags or []
        skill.enabled = True
        skill.skill_path = None
        return skill

    def test_matches_name(self):
        skill = self._make_skill(name="code_executor")
        score = skill.matches("run code_executor on this")
        assert score >= 0.9

    def test_matches_tag(self):
        skill = self._make_skill(tags=["python", "code"])
        score = skill.matches("write some python")
        assert score >= 0.7

    def test_matches_description_keyword(self):
        skill = self._make_skill(description="Search the web for information")
        score = skill.matches("search for docs")
        assert 0 < score <= 0.6

    def test_no_match_returns_zero(self):
        skill = self._make_skill(name="xyz", description="abc", tags=[])
        assert skill.matches("totally unrelated query") == 0.0


class TestBaseSkillFrontmatter:
    def test_parse_skill_md(self, tmp_path):
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text(textwrap.dedent("""\
            ---
            name: custom_tool
            description: Custom description
            version: "3.0"
            author: Dev
            tags: ["custom", "tool"]
            ---
            # Custom Tool
        """))

        class FMSkill(BaseSkill):
            async def execute(self, **kw):
                return {}

        skill = FMSkill(skill_path=tmp_path)
        assert skill.name == "custom_tool"
        assert skill.description == "Custom description"
        assert skill.version == "3.0"
        assert skill.author == "Dev"
        assert skill.tags == ["custom", "tool"]

    def test_no_skill_md(self, tmp_path):
        class NoMD(BaseSkill):
            name = "fallback"

            async def execute(self, **kw):
                return {}

        skill = NoMD(skill_path=tmp_path)
        assert skill.name == "fallback"


# ── SkillLoader ──────────────────────────────────────────────


class TestSkillLoader:
    def _make_skill_dir(self, parent, name, skill_code, skill_md=None):
        """Create a minimal skill directory structure."""
        d = parent / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "skill.py").write_text(skill_code)
        if skill_md:
            (d / "SKILL.md").write_text(skill_md)
        return d

    def test_load_from_directory(self, tmp_path):
        code = textwrap.dedent("""\
            from openclaw.skills.base import BaseSkill

            class MySkill(BaseSkill):
                name = "my_skill"
                description = "My test skill"

                async def execute(self, **kw):
                    return {"success": True}
        """)
        self._make_skill_dir(tmp_path, "my_skill", code)

        loader = SkillLoader.__new__(SkillLoader)
        loader.skills = {}
        loader.settings = MagicMock()

        loader._load_from_directory(tmp_path)
        assert "my_skill" in loader.skills
        assert loader.skills["my_skill"].description == "My test skill"

    def test_invalid_skill_directory_skipped(self, tmp_path):
        """A directory without skill.py is silently skipped."""
        (tmp_path / "empty_skill").mkdir()

        loader = SkillLoader.__new__(SkillLoader)
        loader.skills = {}
        loader.settings = MagicMock()

        loader._load_from_directory(tmp_path)
        assert len(loader.skills) == 0

    def test_skill_with_manifest(self, tmp_path):
        code = textwrap.dedent("""\
            from openclaw.skills.base import BaseSkill

            class ManifestSkill(BaseSkill):
                async def execute(self, **kw):
                    return {}
        """)
        md = textwrap.dedent("""\
            ---
            name: manifest_skill
            description: Has a SKILL.md
            version: "2.0"
            tags: ["test"]
            ---
            # Manifest Skill
        """)
        self._make_skill_dir(tmp_path, "manifest_skill", code, md)

        loader = SkillLoader.__new__(SkillLoader)
        loader.skills = {}
        loader.settings = MagicMock()

        loader._load_from_directory(tmp_path)
        skill = loader.skills["manifest_skill"]
        assert skill.version == "2.0"
        assert skill.tags == ["test"]

    def test_get_skill(self, tmp_path):
        code = textwrap.dedent("""\
            from openclaw.skills.base import BaseSkill

            class SkA(BaseSkill):
                name = "sk_a"
                async def execute(self, **kw):
                    return {}
        """)
        self._make_skill_dir(tmp_path, "sk_a", code)

        loader = SkillLoader.__new__(SkillLoader)
        loader.skills = {}
        loader.settings = MagicMock()
        loader._load_from_directory(tmp_path)

        assert loader.get_skill("sk_a") is not None
        assert loader.get_skill("nonexistent") is None

    def test_list_skills(self, tmp_path):
        code = textwrap.dedent("""\
            from openclaw.skills.base import BaseSkill

            class SkB(BaseSkill):
                name = "sk_b"
                description = "Skill B"
                async def execute(self, **kw):
                    return {}
        """)
        self._make_skill_dir(tmp_path, "sk_b", code)

        loader = SkillLoader.__new__(SkillLoader)
        loader.skills = {}
        loader.settings = MagicMock()
        loader._load_from_directory(tmp_path)

        skills_list = loader.list_skills()
        assert len(skills_list) == 1
        assert skills_list[0]["name"] == "sk_b"

    def test_underscore_dirs_ignored(self, tmp_path):
        """Directories starting with _ are skipped."""
        code = textwrap.dedent("""\
            from openclaw.skills.base import BaseSkill
            class Hidden(BaseSkill):
                name = "hidden"
                async def execute(self, **kw):
                    return {}
        """)
        self._make_skill_dir(tmp_path, "_hidden", code)

        loader = SkillLoader.__new__(SkillLoader)
        loader.skills = {}
        loader.settings = MagicMock()
        loader._load_from_directory(tmp_path)
        assert len(loader.skills) == 0


# ── SkillRouter ──────────────────────────────────────────────


class TestSkillRouter:
    def _make_router(self):
        """Create a router with mock skills."""
        loader = SkillLoader.__new__(SkillLoader)
        loader.skills = {}
        loader.settings = MagicMock()

        class CodeSkill(BaseSkill):
            name = "code_executor"
            description = "Execute code"
            tags = ["code", "python"]

            async def execute(self, **kw):
                return {"success": True, "output": "42"}

        class FileSkill(BaseSkill):
            name = "file_manager"
            description = "Manage files"
            tags = ["file", "read"]

            async def execute(self, **kw):
                return {"success": True}

        loader.skills["code_executor"] = CodeSkill()
        loader.skills["file_manager"] = FileSkill()
        return SkillRouter(loader=loader)

    def test_route_to_correct_skill_by_name(self):
        router = self._make_router()
        skill = router.route("run code_executor")
        assert skill is not None
        assert skill.name == "code_executor"

    def test_route_by_tag(self):
        router = self._make_router()
        skill = router.route("write some python")
        assert skill is not None
        assert skill.name == "code_executor"

    def test_route_unknown_returns_none(self):
        router = self._make_router()
        skill = router.route("play music")
        assert skill is None

    def test_list_available_skills(self):
        router = self._make_router()
        skills = router.list_skills()
        names = [s["name"] for s in skills]
        assert "code_executor" in names
        assert "file_manager" in names

    @pytest.mark.asyncio
    async def test_execute_delegates_to_skill(self):
        router = self._make_router()
        result = await router.execute("run code_executor", code="print(42)")
        assert result is not None
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_no_match_returns_none(self):
        router = self._make_router()
        result = await router.execute("play music")
        assert result is None

    def test_get_skills_description(self):
        router = self._make_router()
        desc = router.get_skills_description()
        assert "code_executor" in desc
        assert "file_manager" in desc

    def test_disabled_skill_not_routed(self):
        router = self._make_router()
        router.loader.skills["code_executor"].enabled = False
        skill = router.route("code_executor")
        assert skill is None or skill.name != "code_executor"


# ── Builtin Skills (mocked I/O) ─────────────────────────────


class TestCodeExecutorSkill:
    @pytest.mark.asyncio
    async def test_python_success(self):
        from openclaw.skills.builtin.code_executor.skill import CodeExecutorSkill

        skill = CodeExecutorSkill()

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"42\n", b""))

        with patch("openclaw.skills.builtin.code_executor.skill.asyncio.create_subprocess_exec",
                    return_value=mock_proc):
            result = await skill.execute(code="print(42)", language="python")

        assert result["success"] is True
        assert "42" in result["stdout"]
        assert result["language"] == "python"

    @pytest.mark.asyncio
    async def test_timeout(self):
        from openclaw.skills.builtin.code_executor.skill import CodeExecutorSkill

        skill = CodeExecutorSkill()

        async def hang(*a, **kw):
            raise asyncio.TimeoutError()

        with patch("openclaw.skills.builtin.code_executor.skill.asyncio.create_subprocess_exec",
                    side_effect=hang):
            result = await skill.execute(code="import time; time.sleep(999)", timeout=1)

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_no_code_returns_error(self):
        from openclaw.skills.builtin.code_executor.skill import CodeExecutorSkill

        skill = CodeExecutorSkill()
        result = await skill.execute(code="")
        assert result["success"] is False
        assert "No code" in result["error"]

    @pytest.mark.asyncio
    async def test_unsupported_language(self):
        from openclaw.skills.builtin.code_executor.skill import CodeExecutorSkill

        skill = CodeExecutorSkill()
        result = await skill.execute(code="x", language="cobol")
        assert result["success"] is False


class TestFileManagerSkill:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        from openclaw.skills.builtin.file_manager.skill import FileManagerSkill

        f = tmp_path / "hello.txt"
        f.write_text("Hello world")

        skill = FileManagerSkill()
        result = await skill.execute(action="read", path=str(f))
        assert result["success"] is True
        assert result["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        from openclaw.skills.builtin.file_manager.skill import FileManagerSkill

        target = tmp_path / "out.txt"

        skill = FileManagerSkill()
        result = await skill.execute(action="write", path=str(target), content="data")
        assert result["success"] is True
        assert target.read_text() == "data"

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self):
        from openclaw.skills.builtin.file_manager.skill import FileManagerSkill

        skill = FileManagerSkill()
        result = await skill.execute(action="read", path="/nonexistent/file.txt")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path):
        from openclaw.skills.builtin.file_manager.skill import FileManagerSkill

        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        skill = FileManagerSkill()
        result = await skill.execute(action="list", path=str(tmp_path))
        assert result["success"] is True
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        from openclaw.skills.builtin.file_manager.skill import FileManagerSkill

        skill = FileManagerSkill()
        result = await skill.execute(action="explode", path=".")
        assert result["success"] is False


class TestSystemInfoSkill:
    @pytest.mark.asyncio
    async def test_returns_valid_dict(self):
        from openclaw.skills.builtin.system_info.skill import SystemInfoSkill

        skill = SystemInfoSkill()
        result = await skill.execute(section="os")
        assert result["success"] is True
        assert "os" in result
        assert "system" in result["os"]

    @pytest.mark.asyncio
    async def test_all_section(self):
        from openclaw.skills.builtin.system_info.skill import SystemInfoSkill

        skill = SystemInfoSkill()
        result = await skill.execute(section="all")
        assert result["success"] is True
        assert "os" in result
        assert "cpu" in result

    @pytest.mark.asyncio
    async def test_env_section(self):
        from openclaw.skills.builtin.system_info.skill import SystemInfoSkill

        skill = SystemInfoSkill()
        result = await skill.execute(section="env")
        assert result["success"] is True
        assert "environment" in result
        assert "cwd" in result["environment"]


class TestWebSearchSkill:
    @pytest.mark.asyncio
    async def test_no_query_returns_error(self):
        from openclaw.skills.builtin.web_search.skill import WebSearchSkill

        skill = WebSearchSkill()
        result = await skill.execute(query="")
        assert result["success"] is False
        assert "No query" in result["error"]

    @pytest.mark.asyncio
    async def test_searxng_formats_results(self):
        from openclaw.skills.builtin.web_search.skill import WebSearchSkill

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Result 1", "url": "http://example.com", "content": "Snippet 1", "engine": "google"},
                {"title": "Result 2", "url": "http://example.org", "content": "Snippet 2", "engine": "bing"},
            ],
            "infoboxes": [],
            "suggestions": [],
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            skill = WebSearchSkill()
            result = await skill.execute(query="test query", max_results=5)

        assert result["success"] is True
        assert result["count"] == 2
        assert result["source"] == "searxng"
        assert result["results"][0]["title"] == "Result 1"

    @pytest.mark.asyncio
    async def test_searxng_failure_falls_back_to_ddg(self):
        from openclaw.skills.builtin.web_search.skill import WebSearchSkill

        # SearXNG fails
        mock_searxng_client = AsyncMock()
        mock_searxng_client.get = AsyncMock(side_effect=Exception("SearXNG down"))
        mock_searxng_client.__aenter__ = AsyncMock(return_value=mock_searxng_client)
        mock_searxng_client.__aexit__ = AsyncMock(return_value=False)

        call_count = 0

        def make_client(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_searxng_client
            # DDG fallback
            mock_ddg = AsyncMock()
            mock_ddg.get = AsyncMock(return_value=MagicMock(
                text='<a rel="nofollow" class="result__a" href="http://ddg.com">DDG Result</a>'
                     '<a class="result__snippet">DDG snippet</a>',
                raise_for_status=MagicMock(),
            ))
            mock_ddg.__aenter__ = AsyncMock(return_value=mock_ddg)
            mock_ddg.__aexit__ = AsyncMock(return_value=False)
            return mock_ddg

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = make_client

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            skill = WebSearchSkill()
            result = await skill.execute(query="test")

        # DDG fallback should have been used
        assert result["source"] == "duckduckgo"
