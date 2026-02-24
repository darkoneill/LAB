"""
Tests for AgentOrchestrator — hierarchical multi-agent delegation.
Covers: _analyze_task, _handle_delegation, max_depth, reset,
        JSON parsing robustness, hierarchy tree, process_message flow.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw.agent.orchestrator import AgentOrchestrator, AgentNode


# ── Helpers ─────────────────────────────────────────────────


def _make_brain_response(content: str, **extra) -> dict:
    return {"content": content, "model": "mock", "usage": {}, "tool_calls": [], **extra}


def _make_analysis_json(should_delegate: bool, subtasks=None, complexity="simple"):
    return json.dumps({
        "should_delegate": should_delegate,
        "complexity": complexity,
        "subtasks": subtasks or [],
        "reasoning": "test",
    })


@pytest.fixture
def fake_settings():
    with patch("openclaw.agent.orchestrator.get_settings") as mock_gs:
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "agent.delegation.enabled": True,
            "agent.delegation.max_depth": 3,
        }.get(k, d)
        settings._base_dir = None
        mock_gs.return_value = settings
        yield settings


@pytest.fixture
def brain(fake_settings):
    b = MagicMock()
    b.generate = AsyncMock(return_value=_make_brain_response("Hello!"))
    return b


@pytest.fixture
def orchestrator(brain):
    return AgentOrchestrator(brain)


# ══════════════════════════════════════════════════════════════
#  Initialization
# ══════════════════════════════════════════════════════════════


class TestInit:

    def test_root_agent_created(self, orchestrator):
        assert orchestrator.root_agent is not None
        assert orchestrator.root_agent.depth == 0
        assert orchestrator.root_agent.role == "primary_assistant"

    def test_root_agent_in_agents_dict(self, orchestrator):
        assert orchestrator.root_agent.id in orchestrator.agents

    def test_max_depth_from_settings(self, orchestrator):
        assert orchestrator.max_depth == 3


# ══════════════════════════════════════════════════════════════
#  _analyze_task — simple → no delegation
# ══════════════════════════════════════════════════════════════


class TestAnalyzeTaskSimple:

    @pytest.mark.asyncio
    async def test_simple_message_no_delegation(self, orchestrator, brain):
        brain.generate = AsyncMock(return_value=_make_brain_response(
            _make_analysis_json(should_delegate=False)
        ))
        analysis = await orchestrator._analyze_task("What time is it?", "")
        assert analysis["should_delegate"] is False

    @pytest.mark.asyncio
    async def test_simple_complexity_tag(self, orchestrator, brain):
        brain.generate = AsyncMock(return_value=_make_brain_response(
            _make_analysis_json(should_delegate=False, complexity="simple")
        ))
        analysis = await orchestrator._analyze_task("hi", "")
        assert analysis["complexity"] == "simple"


# ══════════════════════════════════════════════════════════════
#  _analyze_task — complex → delegation with subtasks
# ══════════════════════════════════════════════════════════════


class TestAnalyzeTaskComplex:

    @pytest.mark.asyncio
    async def test_complex_message_delegation(self, orchestrator, brain):
        brain.generate = AsyncMock(return_value=_make_brain_response(
            _make_analysis_json(
                should_delegate=True,
                subtasks=["research topic", "write code", "run tests"],
                complexity="complex",
            )
        ))
        analysis = await orchestrator._analyze_task("Build me a web app", "")
        assert analysis["should_delegate"] is True
        assert len(analysis["subtasks"]) == 3

    @pytest.mark.asyncio
    async def test_subtasks_preserved(self, orchestrator, brain):
        brain.generate = AsyncMock(return_value=_make_brain_response(
            _make_analysis_json(should_delegate=True, subtasks=["A", "B"])
        ))
        analysis = await orchestrator._analyze_task("complex task", "")
        assert analysis["subtasks"] == ["A", "B"]


# ══════════════════════════════════════════════════════════════
#  JSON parsing robustness — malformed LLM responses
# ══════════════════════════════════════════════════════════════


class TestJSONParsing:

    @pytest.mark.asyncio
    async def test_json_wrapped_in_text(self, orchestrator, brain):
        """JSON embedded in surrounding text should be extracted."""
        content = 'Here is my analysis:\n{"should_delegate": true, "complexity": "complex", "subtasks": ["a"], "reasoning": "yes"}\nDone.'
        brain.generate = AsyncMock(return_value=_make_brain_response(content))
        analysis = await orchestrator._analyze_task("task", "")
        assert analysis["should_delegate"] is True

    @pytest.mark.asyncio
    async def test_completely_invalid_json(self, orchestrator, brain):
        """Total garbage → safe fallback, no exception."""
        brain.generate = AsyncMock(return_value=_make_brain_response(
            "I don't know how to respond in JSON"
        ))
        analysis = await orchestrator._analyze_task("task", "")
        assert analysis["should_delegate"] is False
        assert analysis["subtasks"] == []

    @pytest.mark.asyncio
    async def test_empty_response(self, orchestrator, brain):
        brain.generate = AsyncMock(return_value=_make_brain_response(""))
        analysis = await orchestrator._analyze_task("task", "")
        assert analysis["should_delegate"] is False

    @pytest.mark.asyncio
    async def test_json_missing_should_delegate(self, orchestrator, brain):
        """JSON without should_delegate key → fallback tries rfind, then safe default."""
        brain.generate = AsyncMock(return_value=_make_brain_response(
            '{"complexity": "simple"}'
        ))
        analysis = await orchestrator._analyze_task("task", "")
        # The rfind fallback will parse the JSON but it lacks should_delegate
        # So it returns whatever was parsed (no should_delegate = falsy)
        assert not analysis.get("should_delegate")

    @pytest.mark.asyncio
    async def test_json_with_markdown_fences(self, orchestrator, brain):
        """JSON wrapped in ```json ... ``` blocks."""
        content = '```json\n{"should_delegate": false, "complexity": "simple", "subtasks": [], "reasoning": "ok"}\n```'
        brain.generate = AsyncMock(return_value=_make_brain_response(content))
        analysis = await orchestrator._analyze_task("task", "")
        assert analysis["should_delegate"] is False


# ══════════════════════════════════════════════════════════════
#  _handle_delegation — subtasks spawn sub-agents
# ══════════════════════════════════════════════════════════════


class TestHandleDelegation:

    @pytest.mark.asyncio
    async def test_three_subtasks_three_agents(self, orchestrator, brain):
        """3 subtasks → 3 sub-agent calls + 1 aggregation call = 4 brain.generate calls."""
        call_count = 0

        async def mock_generate(**kwargs):
            nonlocal call_count
            call_count += 1
            return _make_brain_response(f"Result for call {call_count}")

        brain.generate = mock_generate

        analysis = {
            "should_delegate": True,
            "subtasks": ["task A", "task B", "task C"],
        }
        result = await orchestrator._handle_delegation(
            orchestrator.root_agent, "original", analysis, [], ""
        )
        # 3 subtask calls + 1 aggregation = 4
        assert call_count == 4
        # 3 children created
        assert len(orchestrator.root_agent.children_ids) == 3

    @pytest.mark.asyncio
    async def test_children_status_completed(self, orchestrator, brain):
        brain.generate = AsyncMock(return_value=_make_brain_response("done"))

        analysis = {"subtasks": ["sub1", "sub2"]}
        await orchestrator._handle_delegation(
            orchestrator.root_agent, "task", analysis, [], ""
        )
        for cid in orchestrator.root_agent.children_ids:
            child = orchestrator.agents[cid]
            assert child.status == "completed"

    @pytest.mark.asyncio
    async def test_empty_subtasks_direct_execution(self, orchestrator, brain):
        """No subtasks → falls back to direct brain.generate."""
        brain.generate = AsyncMock(return_value=_make_brain_response("direct"))
        analysis = {"subtasks": []}
        result = await orchestrator._handle_delegation(
            orchestrator.root_agent, "task", analysis, [{"role": "user", "content": "task"}], ""
        )
        assert result["content"] == "direct"
        # No children created
        assert len(orchestrator.root_agent.children_ids) == 0

    @pytest.mark.asyncio
    async def test_aggregation_includes_subtask_results(self, orchestrator, brain):
        """The aggregation call should receive subtask results in the prompt."""
        calls = []

        async def mock_generate(**kwargs):
            calls.append(kwargs.get("messages", []))
            return _make_brain_response("aggregated")

        brain.generate = mock_generate
        analysis = {"subtasks": ["sub1"]}
        await orchestrator._handle_delegation(
            orchestrator.root_agent, "original task", analysis, [], ""
        )
        # Last call is aggregation — should contain "Subtask" text
        last_messages = calls[-1]
        agg_content = " ".join(m["content"] for m in last_messages)
        assert "Subtask" in agg_content or "subtask" in agg_content.lower()


# ══════════════════════════════════════════════════════════════
#  max_depth respected
# ══════════════════════════════════════════════════════════════


class TestMaxDepth:

    @pytest.mark.asyncio
    async def test_max_depth_prevents_delegation(self, orchestrator, brain):
        """When root_agent.depth >= max_depth, no delegation occurs."""
        orchestrator.root_agent.depth = orchestrator.max_depth  # at limit

        brain.generate = AsyncMock(side_effect=[
            # _analyze_task returns should_delegate=true
            _make_brain_response(_make_analysis_json(True, ["sub"])),
            # But since depth >= max_depth, direct generate is called
            _make_brain_response("direct response"),
        ])

        result = await orchestrator.process_message("complex task")
        assert result["content"] == "direct response"
        # No children created
        assert len(orchestrator.root_agent.children_ids) == 0


# ══════════════════════════════════════════════════════════════
#  reset()
# ══════════════════════════════════════════════════════════════


class TestReset:

    def test_reset_clears_hierarchy(self, orchestrator):
        # Add some fake children
        child = AgentNode(name="child", depth=1, parent_id=orchestrator.root_agent.id)
        orchestrator.agents[child.id] = child
        orchestrator.root_agent.children_ids.append(child.id)
        assert len(orchestrator.agents) == 2

        orchestrator.reset()
        # Should have only the new root agent
        assert len(orchestrator.agents) == 1
        assert orchestrator.root_agent.children_ids == []

    def test_reset_creates_new_root(self, orchestrator):
        old_id = orchestrator.root_agent.id
        orchestrator.reset()
        assert orchestrator.root_agent.id != old_id


# ══════════════════════════════════════════════════════════════
#  process_message — end-to-end
# ══════════════════════════════════════════════════════════════


class TestProcessMessage:

    @pytest.mark.asyncio
    async def test_simple_message_direct(self, orchestrator, brain):
        """Simple message → analyze says no delegate → direct generate."""
        brain.generate = AsyncMock(side_effect=[
            # _analyze_task
            _make_brain_response(_make_analysis_json(False)),
            # direct generate
            _make_brain_response("Hello user!"),
        ])
        result = await orchestrator.process_message("hi")
        assert result["content"] == "Hello user!"
        assert orchestrator.root_agent.status == "completed"

    @pytest.mark.asyncio
    async def test_delegation_disabled(self, orchestrator, brain, fake_settings):
        """When delegation is disabled, always direct."""
        fake_settings.get = lambda k, d=None: {
            "agent.delegation.enabled": False,
            "agent.delegation.max_depth": 3,
        }.get(k, d)

        brain.generate = AsyncMock(return_value=_make_brain_response("direct"))
        result = await orchestrator.process_message("complex task")
        assert result["content"] == "direct"
        # Only 1 call (no analysis)
        assert brain.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_root_status_set_to_working(self, orchestrator, brain):
        brain.generate = AsyncMock(side_effect=[
            _make_brain_response(_make_analysis_json(False)),
            _make_brain_response("done"),
        ])
        # Before
        assert orchestrator.root_agent.status == "idle"
        await orchestrator.process_message("task")
        # After
        assert orchestrator.root_agent.status == "completed"


# ══════════════════════════════════════════════════════════════
#  get_hierarchy
# ══════════════════════════════════════════════════════════════


class TestHierarchy:

    def test_hierarchy_tree_structure(self, orchestrator):
        tree = orchestrator.get_hierarchy()
        assert tree["id"] == orchestrator.root_agent.id
        assert tree["children"] == []
        assert tree["role"] == "primary_assistant"

    @pytest.mark.asyncio
    async def test_hierarchy_with_children(self, orchestrator, brain):
        brain.generate = AsyncMock(return_value=_make_brain_response("done"))
        analysis = {"subtasks": ["a", "b"]}
        await orchestrator._handle_delegation(
            orchestrator.root_agent, "task", analysis, [], ""
        )
        tree = orchestrator.get_hierarchy()
        assert len(tree["children"]) == 2


# ══════════════════════════════════════════════════════════════
#  _format_subtask_results
# ══════════════════════════════════════════════════════════════


class TestFormatResults:

    def test_formats_numbered_subtasks(self, orchestrator):
        results = [
            {"subtask": "Research", "result": "Found info"},
            {"subtask": "Code", "result": "def foo(): pass"},
        ]
        formatted = orchestrator._format_subtask_results(results)
        assert "Subtask 1" in formatted
        assert "Subtask 2" in formatted
        assert "Research" in formatted
        assert "def foo()" in formatted
