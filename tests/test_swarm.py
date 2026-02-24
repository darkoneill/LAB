"""
Tests for SwarmOrchestrator — Coder→Reviewer→Critic pipeline.
Covers: nominal flow (APPROVED), coder-reviewer loop, ROUTE: routing,
        dry_run rejection, max_iterations, disabled mode, inject_hint,
        _parse_routing, _compress_feedback, _strip_code_fences.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw.agent.swarm import (
    SwarmOrchestrator,
    SwarmResult,
    AgentRole,
    AGENT_PROFILES,
    ROUTE_MAPPING,
)


# ── Helpers ─────────────────────────────────────────────────


def _resp(content: str) -> dict:
    return {"content": content, "model": "mock", "usage": {}, "tool_calls": []}


@pytest.fixture
def fake_settings():
    with patch("openclaw.agent.swarm.get_settings") as mock_gs:
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "agent.swarm.enabled": True,
            "agent.swarm.max_iterations": 3,
            "agent.swarm.dry_run": False,  # Disabled by default in tests
            "sandbox.workspace_path": "/nonexistent",
        }.get(k, d)
        mock_gs.return_value = settings
        yield settings


@pytest.fixture
def brain(fake_settings):
    b = MagicMock()
    b.generate = AsyncMock(return_value=_resp("ok"))
    return b


@pytest.fixture
def swarm(brain):
    with patch("openclaw.agent.swarm.generate_repo_map", return_value=""):
        return SwarmOrchestrator(brain)


# ══════════════════════════════════════════════════════════════
#  Nominal flow: code → review APPROVED
# ══════════════════════════════════════════════════════════════


class TestNominalApproved:

    @pytest.mark.asyncio
    async def test_code_approved_first_iteration(self, swarm, brain):
        brain.generate = AsyncMock(side_effect=[
            _resp("def hello(): return 'hi'"),  # Coder
            _resp("APPROVED"),                   # Reviewer
        ])
        result = await swarm.execute_swarm("write hello func")
        assert result.success is True
        assert result.code == "def hello(): return 'hi'"
        assert result.iterations == 1
        assert AgentRole.CODER in result.agents_used
        assert AgentRole.REVIEWER in result.agents_used

    @pytest.mark.asyncio
    async def test_approved_stops_loop(self, swarm, brain):
        brain.generate = AsyncMock(side_effect=[
            _resp("code v1"),     # Coder iter 1
            _resp("APPROVED"),    # Reviewer iter 1 → APPROVED → stop
        ])
        result = await swarm.execute_swarm("task")
        assert result.iterations == 1
        assert brain.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_approved_case_insensitive(self, swarm, brain):
        brain.generate = AsyncMock(side_effect=[
            _resp("code"),
            _resp("Everything looks good. Approved."),  # lowercase mixed
        ])
        result = await swarm.execute_swarm("task")
        assert result.iterations == 1


# ══════════════════════════════════════════════════════════════
#  Coder-Reviewer loop: corrections needed
# ══════════════════════════════════════════════════════════════


class TestCoderReviewerLoop:

    @pytest.mark.asyncio
    async def test_one_correction_then_approved(self, swarm, brain):
        brain.generate = AsyncMock(side_effect=[
            _resp("code v1"),                         # Coder iter 1
            _resp("MAJEUR: missing error handling"),   # Reviewer iter 1 → NOT approved
            _resp("code v2 (fixed)"),                  # Coder iter 2
            _resp("APPROVED"),                         # Reviewer iter 2 → APPROVED
        ])
        result = await swarm.execute_swarm("task")
        assert result.iterations == 2
        assert result.code == "code v2 (fixed)"
        assert brain.generate.call_count == 4

    @pytest.mark.asyncio
    async def test_review_feedback_passed_to_coder(self, swarm, brain):
        """On iter 2, coder receives the reviewer's feedback."""
        calls = []

        async def capture(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _resp("code v1")
            elif len(calls) == 2:
                return _resp("Fix the bug please")
            elif len(calls) == 3:
                return _resp("code v2")
            else:
                return _resp("APPROVED")

        brain.generate = capture
        await swarm.execute_swarm("task")
        # Call 3 is coder iter 2 — messages should contain feedback
        coder_iter2_msg = calls[2]["messages"][0]["content"]
        assert "Fix the bug please" in coder_iter2_msg


# ══════════════════════════════════════════════════════════════
#  Conditional routing: ROUTE:security
# ══════════════════════════════════════════════════════════════


class TestRouting:

    @pytest.mark.asyncio
    async def test_route_security_triggers_agent(self, swarm, brain):
        brain.generate = AsyncMock(side_effect=[
            _resp("vulnerable code"),              # Coder iter 1
            _resp("CRITIQUE: SQL injection\nROUTE:security"),  # Reviewer
            _resp("[VULN-001] SQL injection found"),            # Security agent
            _resp("safe code v2"),                              # Coder iter 2
            _resp("APPROVED"),                                  # Reviewer
        ])
        result = await swarm.execute_swarm("task")
        assert AgentRole.SECURITY in result.agents_used
        assert result.iterations == 2

    def test_parse_routing_single(self):
        output = "MAJEUR: XSS possible\nROUTE:security\nDetails..."
        roles = SwarmOrchestrator._parse_routing(output)
        assert roles == [AgentRole.SECURITY]

    def test_parse_routing_multiple(self):
        output = "ROUTE:security\nROUTE:tester"
        roles = SwarmOrchestrator._parse_routing(output)
        assert AgentRole.SECURITY in roles
        assert AgentRole.TESTER in roles

    def test_parse_routing_unknown_ignored(self):
        output = "ROUTE:unknown_agent"
        roles = SwarmOrchestrator._parse_routing(output)
        assert roles == []

    def test_parse_routing_no_routes(self):
        output = "Everything is fine. APPROVED"
        roles = SwarmOrchestrator._parse_routing(output)
        assert roles == []

    def test_parse_routing_deduplicates(self):
        output = "ROUTE:security\nROUTE:security"
        roles = SwarmOrchestrator._parse_routing(output)
        assert len(roles) == 1


# ══════════════════════════════════════════════════════════════
#  Dry run — SyntaxError rejection
# ══════════════════════════════════════════════════════════════


class TestDryRun:

    @pytest.mark.asyncio
    async def test_dry_run_detects_syntax_error(self, brain, fake_settings):
        """When dry_run is enabled and code has SyntaxError, trace contains dry_run entry."""
        fake_settings.get = lambda k, d=None: {
            "agent.swarm.enabled": True,
            "agent.swarm.max_iterations": 2,
            "agent.swarm.dry_run": True,
            "sandbox.workspace_path": "/nonexistent",
        }.get(k, d)

        brain.generate = AsyncMock(side_effect=[
            _resp("def broken(\n"),     # Coder: syntax error
            _resp("APPROVED"),           # Reviewer (still gets called)
        ])
        with patch("openclaw.agent.swarm.generate_repo_map", return_value=""):
            swarm = SwarmOrchestrator(brain)

        result = await swarm.execute_swarm("task")
        dry_run_entries = [t for t in result.trace if t.get("phase") == "dry_run"]
        assert len(dry_run_entries) >= 1
        assert "ERREUR" in dry_run_entries[0]["output"]

    @pytest.mark.asyncio
    async def test_dry_run_clean_code_no_error(self, brain, fake_settings):
        fake_settings.get = lambda k, d=None: {
            "agent.swarm.enabled": True,
            "agent.swarm.max_iterations": 1,
            "agent.swarm.dry_run": True,
            "sandbox.workspace_path": "/nonexistent",
        }.get(k, d)

        brain.generate = AsyncMock(side_effect=[
            _resp("x = 42\nprint(x)"),  # Coder: valid code
            _resp("APPROVED"),
        ])
        with patch("openclaw.agent.swarm.generate_repo_map", return_value=""):
            swarm = SwarmOrchestrator(brain)

        result = await swarm.execute_swarm("task")
        dry_run_entries = [t for t in result.trace if t.get("phase") == "dry_run"]
        # No error entry: dry_run_compile returns "" for valid code
        assert len(dry_run_entries) == 0


# ══════════════════════════════════════════════════════════════
#  max_iterations — loop bound
# ══════════════════════════════════════════════════════════════


class TestMaxIterations:

    @pytest.mark.asyncio
    async def test_loop_stops_at_max(self, swarm, brain):
        """Reviewer never approves → loop stops at max_iterations (3)."""
        brain.generate = AsyncMock(return_value=_resp("NOT APPROVED: issues remain"))
        # Override coder to return different code each time
        call_idx = 0

        async def alternating(**kwargs):
            nonlocal call_idx
            call_idx += 1
            if call_idx % 2 == 1:
                return _resp(f"code v{call_idx}")
            return _resp("CRITIQUE: still broken")

        brain.generate = alternating
        result = await swarm.execute_swarm("task")
        assert result.iterations == swarm._max_iterations

    @pytest.mark.asyncio
    async def test_single_iteration_max(self, brain, fake_settings):
        fake_settings.get = lambda k, d=None: {
            "agent.swarm.enabled": True,
            "agent.swarm.max_iterations": 1,
            "agent.swarm.dry_run": False,
            "sandbox.workspace_path": "/nonexistent",
        }.get(k, d)

        brain.generate = AsyncMock(side_effect=[
            _resp("code"),
            _resp("NOT approved"),
        ])
        with patch("openclaw.agent.swarm.generate_repo_map", return_value=""):
            swarm = SwarmOrchestrator(brain)
        result = await swarm.execute_swarm("task")
        assert result.iterations == 1


# ══════════════════════════════════════════════════════════════
#  Disabled mode
# ══════════════════════════════════════════════════════════════


class TestDisabled:

    @pytest.mark.asyncio
    async def test_disabled_fallback_to_single_agent(self, brain, fake_settings):
        fake_settings.get = lambda k, d=None: {
            "agent.swarm.enabled": False,
            "agent.swarm.max_iterations": 3,
            "agent.swarm.dry_run": False,
            "sandbox.workspace_path": "/nonexistent",
        }.get(k, d)

        brain.generate = AsyncMock(return_value=_resp("single agent response"))
        with patch("openclaw.agent.swarm.generate_repo_map", return_value=""):
            swarm = SwarmOrchestrator(brain)
        result = await swarm.execute_swarm("task")
        assert result.success is True
        assert result.final_output == "single agent response"
        assert result.agents_used == ["default"]


# ══════════════════════════════════════════════════════════════
#  Critic validation
# ══════════════════════════════════════════════════════════════


class TestCritic:

    @pytest.mark.asyncio
    async def test_critic_validates(self, swarm, brain):
        brain.generate = AsyncMock(side_effect=[
            _resp("good code"),    # Coder
            _resp("APPROVED"),     # Reviewer
            _resp("VALIDE"),       # Critic
        ])
        result = await swarm.execute_swarm(
            "task", roles=[AgentRole.CODER, AgentRole.REVIEWER, AgentRole.CRITIC]
        )
        assert result.validated is True
        assert AgentRole.CRITIC in result.agents_used

    @pytest.mark.asyncio
    async def test_critic_rejects(self, swarm, brain):
        brain.generate = AsyncMock(side_effect=[
            _resp("code"),
            _resp("APPROVED"),
            _resp("[ERREUR] hallucination detected"),  # Critic rejects
        ])
        result = await swarm.execute_swarm(
            "task", roles=[AgentRole.CODER, AgentRole.REVIEWER, AgentRole.CRITIC]
        )
        assert result.validated is False


# ══════════════════════════════════════════════════════════════
#  inject_hint
# ══════════════════════════════════════════════════════════════


class TestInjectHint:

    @pytest.mark.asyncio
    async def test_hint_consumed_by_coder(self, swarm, brain):
        swarm.inject_hint("Use asyncio please")
        calls = []

        async def capture(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _resp("code with asyncio")
            return _resp("APPROVED")

        brain.generate = capture
        await swarm.execute_swarm("task")
        # First coder call should contain the hint
        coder_msg = calls[0]["messages"][0]["content"]
        assert "Use asyncio please" in coder_msg

    @pytest.mark.asyncio
    async def test_hint_cleared_after_use(self, swarm, brain):
        swarm.inject_hint("hint text")
        brain.generate = AsyncMock(side_effect=[
            _resp("code"),
            _resp("APPROVED"),
        ])
        await swarm.execute_swarm("task")
        assert swarm._pending_hint == ""


# ══════════════════════════════════════════════════════════════
#  Utility methods
# ══════════════════════════════════════════════════════════════


class TestUtilities:

    def test_agent_profiles_defined(self):
        for role in AgentRole:
            assert role in AGENT_PROFILES

    def test_route_mapping_targets_valid_roles(self):
        for target in ROUTE_MAPPING.values():
            assert target in AGENT_PROFILES

    def test_get_available_profiles(self, swarm):
        profiles = swarm.get_available_profiles()
        assert "coder" in profiles
        assert "reviewer" in profiles
        assert "name" in profiles["coder"]

    def test_get_active_agents_empty(self, swarm):
        assert swarm.get_active_agents() == []


# ══════════════════════════════════════════════════════════════
#  Planner phase
# ══════════════════════════════════════════════════════════════


class TestPlanner:

    @pytest.mark.asyncio
    async def test_planner_enriches_task(self, swarm, brain):
        calls = []

        async def capture(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _resp("Step 1: do A. Step 2: do B.")
            elif len(calls) == 2:
                return _resp("implemented code")
            else:
                return _resp("APPROVED")

        brain.generate = capture
        result = await swarm.execute_swarm(
            "build feature X",
            roles=[AgentRole.PLANNER, AgentRole.CODER, AgentRole.REVIEWER],
        )
        assert AgentRole.PLANNER in result.agents_used
        # Coder call (call index 1) should contain the plan
        coder_msg = calls[1]["messages"][0]["content"]
        assert "Step 1" in coder_msg


# ══════════════════════════════════════════════════════════════
#  Trace recording
# ══════════════════════════════════════════════════════════════


class TestTrace:

    @pytest.mark.asyncio
    async def test_trace_records_phases(self, swarm, brain):
        brain.generate = AsyncMock(side_effect=[
            _resp("code"),
            _resp("APPROVED"),
        ])
        result = await swarm.execute_swarm("task")
        phases = [t["phase"] for t in result.trace]
        assert "coding" in phases
        assert "review" in phases
