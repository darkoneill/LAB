"""
Tests for openclaw/agent/brain.py
Mocks all LLM calls (Anthropic + OpenAI) with AsyncMock.
Covers: system message, tool definitions, tool execution routing,
        provider failover, tool_call parsing, CoT extraction, context trimming.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw.agent.brain import AgentBrain, COT_INSTRUCTION


# ── Fixtures ────────────────────────────────────────────────


class FakeSettings:
    _base_dir = None

    def __init__(self, overrides: dict = None):
        self._data = {
            "agent.temperature": 0.7,
            "agent.max_tokens": 4096,
            "agent.max_iterations": 25,
            "agent.chain_of_thought": True,
            "agent.system_prompt_file": "agent/prompts/system.md",
            "agent.personality_file": "agent/prompts/personality.md",
            "agent.tools_prompt_file": "agent/prompts/tools.md",
            "providers.anthropic.enabled": True,
            "providers.anthropic.api_key": "sk-test-key",
            "providers.anthropic.default_model": "claude-sonnet-4-20250514",
            "providers.anthropic.models": [
                {"id": "claude-sonnet-4-20250514", "name": "Claude Sonnet"},
            ],
            "providers.openai.enabled": True,
            "providers.openai.api_key": "sk-openai-test",
            "providers.openai.default_model": "gpt-4o",
            "providers.openai.base_url": None,
            "providers.openai.models": [
                {"id": "gpt-4o", "name": "GPT-4o"},
            ],
            "providers.ollama.enabled": False,
            "providers.custom.enabled": False,
        }
        if overrides:
            self._data.update(overrides)

    def get(self, dotpath, default=None):
        return self._data.get(dotpath, default)


def _make_anthropic_response(text="Hello!", tool_use_blocks=None, input_tokens=10, output_tokens=20):
    """Build a fake Anthropic messages.create() response."""
    blocks = []
    if text:
        blocks.append(SimpleNamespace(type="text", text=text))
    for tu in (tool_use_blocks or []):
        blocks.append(SimpleNamespace(
            type="tool_use",
            id=tu["id"],
            name=tu["name"],
            input=tu.get("input", {}),
        ))
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _make_openai_response(text="Hello!", tool_calls=None, prompt_tokens=10, completion_tokens=20):
    """Build a fake OpenAI chat.completions.create() response."""
    tc_objects = None
    if tool_calls:
        tc_objects = []
        for tc in tool_calls:
            tc_objects.append(SimpleNamespace(
                id=tc["id"],
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=tc.get("arguments", "{}"),
                ),
            ))
    choice = SimpleNamespace(
        message=SimpleNamespace(
            content=text,
            tool_calls=tc_objects,
        ),
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


@pytest.fixture
def fake_settings():
    return FakeSettings()


@pytest.fixture
def tool_executor():
    """Mock ToolExecutor."""
    te = MagicMock()
    te.execute = AsyncMock(return_value={"success": True, "stdout": "ok"})
    return te


@pytest.fixture
def skill_router():
    """Mock SkillRouter with one skill."""
    mock_skill = AsyncMock()
    mock_skill.execute = AsyncMock(return_value={"success": True, "results": []})

    loader = MagicMock()
    loader.get_skill = MagicMock(return_value=mock_skill)

    sr = MagicMock()
    sr.list_skills.return_value = [
        {"name": "web_search", "description": "Search the web"},
        {"name": "code_executor", "description": "Execute code"},
    ]
    sr.get_skills_description.return_value = "- **web_search**: Search the web"
    sr.loader = loader
    return sr


@pytest.fixture
def brain(fake_settings, tool_executor, skill_router):
    """AgentBrain with mocked settings and no prompt files."""
    with patch("openclaw.agent.brain.get_settings", return_value=fake_settings), \
         patch("openclaw.gateway.router.get_settings", return_value=fake_settings):
        b = AgentBrain(tool_executor=tool_executor, skill_router=skill_router)
    return b


@pytest.fixture
def brain_no_tools(fake_settings):
    """AgentBrain without tool_executor or skill_router."""
    with patch("openclaw.agent.brain.get_settings", return_value=fake_settings), \
         patch("openclaw.gateway.router.get_settings", return_value=fake_settings):
        b = AgentBrain(tool_executor=None, skill_router=None)
    return b


# ══════════════════════════════════════════════════════════════
#  SYSTEM MESSAGE CONSTRUCTION
# ══════════════════════════════════════════════════════════════


class TestSystemMessage:

    def test_includes_cot_instruction(self, brain):
        msg = brain._build_system_message()
        assert "Reasoning Protocol" in msg
        assert "<thinking>" in msg

    def test_cot_disabled(self, brain, fake_settings):
        fake_settings._data["agent.chain_of_thought"] = False
        msg = brain._build_system_message()
        assert "Reasoning Protocol" not in msg

    def test_includes_memory_context(self, brain):
        msg = brain._build_system_message(memory_context="User prefers dark mode")
        assert "Relevant Memories" in msg
        assert "dark mode" in msg

    def test_no_memory_section_if_empty(self, brain):
        msg = brain._build_system_message(memory_context="")
        assert "Relevant Memories" not in msg

    def test_includes_skills_description(self, brain):
        msg = brain._build_system_message()
        assert "Available Skills" in msg
        assert "web_search" in msg

    def test_no_skills_section_without_router(self, brain_no_tools):
        msg = brain_no_tools._build_system_message()
        assert "Available Skills" not in msg

    def test_includes_personality(self, brain):
        brain._personality = "I am helpful and concise."
        msg = brain._build_system_message()
        assert "Personality" in msg
        assert "helpful and concise" in msg

    def test_includes_tools_prompt(self, brain):
        brain._tools_prompt = "shell: run commands"
        msg = brain._build_system_message()
        assert "Available Tools" in msg
        assert "shell: run commands" in msg


# ══════════════════════════════════════════════════════════════
#  TOOL DEFINITIONS
# ══════════════════════════════════════════════════════════════


class TestToolDefinitions:

    def test_anthropic_format_default_tools(self, brain):
        tools = brain._get_tool_definitions_anthropic()
        names = {t["name"] for t in tools}
        assert "shell" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "search_files" in names

    def test_anthropic_format_includes_skills(self, brain):
        tools = brain._get_tool_definitions_anthropic()
        names = {t["name"] for t in tools}
        assert "skill_web_search" in names
        assert "skill_code_executor" in names

    def test_anthropic_tool_schema_structure(self, brain):
        tools = brain._get_tool_definitions_anthropic()
        shell = next(t for t in tools if t["name"] == "shell")
        assert "input_schema" in shell
        assert shell["input_schema"]["type"] == "object"
        assert "command" in shell["input_schema"]["properties"]
        assert "command" in shell["input_schema"]["required"]

    def test_openai_format_wraps_anthropic(self, brain):
        tools = brain._get_tool_definitions_openai()
        assert all(t["type"] == "function" for t in tools)
        shell = next(t for t in tools if t["function"]["name"] == "shell")
        assert "parameters" in shell["function"]
        assert shell["function"]["parameters"]["type"] == "object"

    def test_openai_format_count_matches(self, brain):
        a_tools = brain._get_tool_definitions_anthropic()
        o_tools = brain._get_tool_definitions_openai()
        assert len(a_tools) == len(o_tools)

    def test_no_tools_without_executor(self, brain_no_tools):
        tools = brain_no_tools._get_tool_definitions_anthropic()
        assert tools == []

    def test_cache_is_reused(self, brain):
        tools1 = brain._get_tool_definitions_anthropic()
        tools2 = brain._get_tool_definitions_anthropic()
        assert tools1 is tools2  # same object reference

    def test_invalidate_tool_cache(self, brain):
        tools_before = brain._get_tool_definitions_anthropic()
        brain.invalidate_tool_cache()
        tools_after = brain._get_tool_definitions_anthropic()
        assert tools_before is not tools_after
        assert tools_before == tools_after  # same content, different object


# ══════════════════════════════════════════════════════════════
#  MESSAGE FORMATTING
# ══════════════════════════════════════════════════════════════


class TestMessageFormatting:

    def test_anthropic_skips_system_role(self, brain):
        messages = [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ]
        system, formatted = brain._format_messages_anthropic(messages, "sys prompt")
        assert system == "sys prompt"
        roles = [m["role"] for m in formatted]
        assert "system" not in roles
        assert formatted[-1]["content"] == "hi"

    def test_anthropic_merges_consecutive_same_role(self, brain):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        _, formatted = brain._format_messages_anthropic(messages, "sys")
        assert len(formatted) == 1
        assert "first" in formatted[0]["content"]
        assert "second" in formatted[0]["content"]

    def test_anthropic_ensures_user_first(self, brain):
        messages = [{"role": "assistant", "content": "I start"}]
        _, formatted = brain._format_messages_anthropic(messages, "sys")
        assert formatted[0]["role"] == "user"

    def test_anthropic_remaps_unknown_role(self, brain):
        messages = [{"role": "tool", "content": "result"}]
        _, formatted = brain._format_messages_anthropic(messages, "sys")
        assert all(m["role"] in ("user", "assistant") for m in formatted)

    def test_anthropic_context_trimming(self, brain):
        """Very long history should be trimmed to ~400k chars."""
        big_msg = "x" * 200_001
        messages = [
            {"role": "user", "content": big_msg},
            {"role": "assistant", "content": big_msg},
            {"role": "user", "content": "latest question"},
        ]
        _, formatted = brain._format_messages_anthropic(messages, "sys")
        total = sum(len(m["content"]) for m in formatted)
        assert total <= 400_001
        # The latest message should be preserved
        assert formatted[-1]["content"] == "latest question"

    def test_openai_prepends_system(self, brain):
        messages = [{"role": "user", "content": "hi"}]
        formatted = brain._format_messages_openai(messages, "sys prompt")
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == "sys prompt"

    def test_openai_remaps_unknown_role(self, brain):
        messages = [{"role": "function", "content": "ok"}]
        formatted = brain._format_messages_openai(messages, "sys")
        assert formatted[1]["role"] == "user"

    # ── Anthropic tool_calls + tool_result formatting ──

    def test_anthropic_assistant_with_tool_calls(self, brain):
        """Assistant message with tool_calls → content blocks (text + tool_use)."""
        messages = [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}}],
            },
        ]
        _, formatted = brain._format_messages_anthropic(messages, "sys")
        assistant_msg = formatted[-1]
        assert assistant_msg["role"] == "assistant"
        assert isinstance(assistant_msg["content"], list)
        assert assistant_msg["content"][0]["type"] == "text"
        assert assistant_msg["content"][0]["text"] == "Let me check."
        assert assistant_msg["content"][1]["type"] == "tool_use"
        assert assistant_msg["content"][1]["id"] == "tu_1"
        assert assistant_msg["content"][1]["name"] == "shell"
        assert assistant_msg["content"][1]["input"]["command"] == "ls"

    def test_anthropic_tool_result_message(self, brain):
        """tool_result role → user message with tool_result content blocks."""
        messages = [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}}],
            },
            {
                "role": "tool_result",
                "tool_results": [{"tool_use_id": "tu_1", "content": "file1.txt\nfile2.txt"}],
            },
        ]
        _, formatted = brain._format_messages_anthropic(messages, "sys")
        tool_result_msg = formatted[-1]
        assert tool_result_msg["role"] == "user"
        assert isinstance(tool_result_msg["content"], list)
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_use_id"] == "tu_1"
        assert "file1.txt" in tool_result_msg["content"][0]["content"]

    def test_anthropic_no_merge_after_blocks(self, brain):
        """Plain text user msg after tool_result blocks should NOT be merged."""
        messages = [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}}],
            },
            {
                "role": "tool_result",
                "tool_results": [{"tool_use_id": "tu_1", "content": "ok"}],
            },
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "thanks"},
        ]
        _, formatted = brain._format_messages_anthropic(messages, "sys")
        # Should have: user, assistant(blocks), user(tool_result blocks), assistant, user
        assert len(formatted) == 5

    # ── OpenAI tool_calls + tool_result formatting ──

    def test_openai_assistant_with_tool_calls(self, brain):
        """Assistant message with tool_calls → OpenAI tool_calls array."""
        messages = [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [{"id": "call_1", "name": "shell", "arguments": {"command": "ls"}}],
            },
        ]
        formatted = brain._format_messages_openai(messages, "sys")
        assistant_msg = formatted[-1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["tool_calls"]) == 1
        tc = assistant_msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "shell"
        assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}

    def test_openai_tool_result_messages(self, brain):
        """tool_result role → one 'tool' message per result."""
        messages = [
            {"role": "user", "content": "list files"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "shell", "arguments": {"command": "ls"}},
                    {"id": "call_2", "name": "read_file", "arguments": {"path": "/tmp/x"}},
                ],
            },
            {
                "role": "tool_result",
                "tool_results": [
                    {"tool_use_id": "call_1", "content": "file.txt"},
                    {"tool_use_id": "call_2", "content": "contents"},
                ],
            },
        ]
        formatted = brain._format_messages_openai(messages, "sys")
        tool_msgs = [m for m in formatted if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert tool_msgs[0]["content"] == "file.txt"
        assert tool_msgs[1]["tool_call_id"] == "call_2"


# ══════════════════════════════════════════════════════════════
#  COT EXTRACTION
# ══════════════════════════════════════════════════════════════


class TestThinkingExtraction:

    def test_extract_thinking(self, brain):
        content = "<thinking>I need to check the file.</thinking>Here is the result."
        thinking, clean = brain._extract_thinking(content)
        assert thinking == "I need to check the file."
        assert clean == "Here is the result."

    def test_extract_multiple_blocks(self, brain):
        content = "<thinking>Step 1</thinking>Middle<thinking>Step 2</thinking>End"
        thinking, clean = brain._extract_thinking(content)
        assert "Step 1" in thinking
        assert "Step 2" in thinking
        assert clean == "MiddleEnd"

    def test_no_thinking_blocks(self, brain):
        content = "Just a plain response."
        thinking, clean = brain._extract_thinking(content)
        assert thinking == ""
        assert clean == "Just a plain response."

    def test_validate_thinking_shell(self, brain):
        assert brain._validate_thinking("This is a safe read-only command", "shell")
        assert brain._validate_thinking("I need to verify this first", "shell")
        assert not brain._validate_thinking("Let me do this", "shell")

    def test_validate_thinking_empty(self, brain):
        assert not brain._validate_thinking("", "shell")
        assert not brain._validate_thinking("", None)

    def test_validate_thinking_non_shell(self, brain):
        assert brain._validate_thinking("Any content here", None)
        assert brain._validate_thinking("Any content here", "read_file")


# ══════════════════════════════════════════════════════════════
#  ANTHROPIC TOOL_CALL PARSING
# ══════════════════════════════════════════════════════════════


class TestAnthropicParsing:

    @pytest.mark.asyncio
    async def test_parse_text_only(self, brain):
        resp = _make_anthropic_response(text="Hello world")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            result = await brain._call_anthropic("claude-sonnet-4-20250514", [{"role": "user", "content": "hi"}], "sys", 0.7, 4096)

        assert result["content"] == "Hello world"
        assert result["tool_calls"] == []
        assert result["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_parse_tool_use_blocks(self, brain):
        resp = _make_anthropic_response(
            text="Let me search.",
            tool_use_blocks=[
                {"id": "tu_1", "name": "shell", "input": {"command": "ls -la"}},
                {"id": "tu_2", "name": "read_file", "input": {"path": "/tmp/test"}},
            ],
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            result = await brain._call_anthropic("claude-sonnet-4-20250514", [{"role": "user", "content": "list files"}], "sys", 0.7, 4096)

        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["name"] == "shell"
        assert result["tool_calls"][0]["arguments"]["command"] == "ls -la"
        assert result["tool_calls"][1]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_passes_tools_when_executor_present(self, brain):
        resp = _make_anthropic_response(text="ok")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            await brain._call_anthropic("model", [{"role": "user", "content": "hi"}], "sys", 0.7, 4096)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert any(t["name"] == "shell" for t in call_kwargs["tools"])

    @pytest.mark.asyncio
    async def test_no_tools_without_executor(self, brain_no_tools):
        resp = _make_anthropic_response(text="ok")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(brain_no_tools, "_get_anthropic_client", return_value=mock_client):
            await brain_no_tools._call_anthropic("model", [{"role": "user", "content": "hi"}], "sys", 0.7, 4096)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs


# ══════════════════════════════════════════════════════════════
#  OPENAI TOOL_CALL PARSING
# ══════════════════════════════════════════════════════════════


class TestOpenAIParsing:

    @pytest.mark.asyncio
    async def test_parse_text_only(self, brain):
        resp = _make_openai_response(text="Hello!")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(brain, "_get_provider_client", return_value=mock_client):
            result = await brain._call_openai_compat("openai", "gpt-4o", [{"role": "user", "content": "hi"}], "sys", 0.7, 4096)

        assert result["content"] == "Hello!"
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_parse_tool_calls_json_args(self, brain):
        resp = _make_openai_response(
            text="",
            tool_calls=[
                {"id": "call_1", "name": "shell", "arguments": '{"command": "ls"}'},
            ],
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(brain, "_get_provider_client", return_value=mock_client):
            result = await brain._call_openai_compat("openai", "gpt-4o", [{"role": "user", "content": "list"}], "sys", 0.7, 4096)

        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["arguments"]["command"] == "ls"

    @pytest.mark.asyncio
    async def test_parse_tool_calls_invalid_json_args(self, brain):
        """If arguments is a non-JSON string, it should be wrapped as {"query": ...}."""
        resp = _make_openai_response(
            text="",
            tool_calls=[
                {"id": "call_1", "name": "skill_web_search", "arguments": "search for cats"},
            ],
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(brain, "_get_provider_client", return_value=mock_client):
            result = await brain._call_openai_compat("openai", "gpt-4o", [{"role": "user", "content": "search"}], "sys", 0.7, 4096)

        assert result["tool_calls"][0]["arguments"] == {"query": "search for cats"}

    @pytest.mark.asyncio
    async def test_no_tools_for_ollama(self, brain, fake_settings):
        """Ollama provider should NOT receive tool definitions."""
        fake_settings._data["providers.ollama.enabled"] = True
        fake_settings._data["providers.ollama.base_url"] = "http://localhost:11434"

        resp = _make_openai_response(text="ok")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(brain, "_get_provider_client", return_value=mock_client):
            await brain._call_openai_compat("ollama", "llama3", [{"role": "user", "content": "hi"}], "sys", 0.7, 4096)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    async def test_passes_tools_for_openai(self, brain):
        resp = _make_openai_response(text="ok")
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=resp)

        with patch.object(brain, "_get_provider_client", return_value=mock_client):
            await brain._call_openai_compat("openai", "gpt-4o", [{"role": "user", "content": "hi"}], "sys", 0.7, 4096)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert any(t["function"]["name"] == "shell" for t in call_kwargs["tools"])


# ══════════════════════════════════════════════════════════════
#  TOOL EXECUTION ROUTING
# ══════════════════════════════════════════════════════════════


class TestToolExecution:

    @pytest.mark.asyncio
    async def test_execute_regular_tool(self, brain, tool_executor):
        results = await brain._execute_tools([
            {"name": "shell", "arguments": {"command": "echo hello"}},
        ])
        assert len(results) == 1
        assert results[0]["success"] is True
        tool_executor.execute.assert_called_once_with("shell", {"command": "echo hello"})

    @pytest.mark.asyncio
    async def test_execute_skill_tool(self, brain, skill_router):
        results = await brain._execute_tools([
            {"name": "skill_web_search", "arguments": {"query": "Python 3.12"}},
        ])
        assert len(results) == 1
        assert results[0]["success"] is True
        skill_router.loader.get_skill.assert_called_once_with("web_search")

    @pytest.mark.asyncio
    async def test_execute_skill_not_found(self, brain, skill_router):
        skill_router.loader.get_skill.return_value = None
        results = await brain._execute_tools([
            {"name": "skill_nonexistent", "arguments": {"query": "test"}},
        ])
        assert results[0]["success"] is True  # the outer wrapper succeeds
        assert results[0]["result"]["success"] is False
        assert "not found" in results[0]["result"]["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_tool_exception(self, brain, tool_executor):
        tool_executor.execute = AsyncMock(side_effect=RuntimeError("boom"))
        results = await brain._execute_tools([
            {"name": "shell", "arguments": {"command": "bad"}},
        ])
        assert results[0]["success"] is False
        assert "boom" in results[0]["error"]

    @pytest.mark.asyncio
    async def test_execute_multiple_tools(self, brain, tool_executor):
        results = await brain._execute_tools([
            {"name": "shell", "arguments": {"command": "ls"}},
            {"name": "read_file", "arguments": {"path": "/tmp/x"}},
        ])
        assert len(results) == 2
        assert tool_executor.execute.call_count == 2


# ══════════════════════════════════════════════════════════════
#  GENERATE — full flow with mocked provider
# ══════════════════════════════════════════════════════════════


class TestGenerate:

    @pytest.mark.asyncio
    async def test_generate_simple_text(self, brain):
        resp = _make_anthropic_response(text="Hello user!")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            result = await brain.generate([{"role": "user", "content": "hi"}])

        assert result["content"] == "Hello user!"
        assert result["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_generate_with_thinking(self, brain):
        resp = _make_anthropic_response(text="<thinking>This is safe.</thinking>Result here.")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            result = await brain.generate([{"role": "user", "content": "do something"}])

        assert result["content"] == "Result here."
        assert result["thinking"] == "This is safe."

    @pytest.mark.asyncio
    async def test_generate_tool_call_loop(self, brain, tool_executor):
        """When LLM returns tool_calls, brain should execute then re-call LLM."""
        # First call: LLM returns a tool_use
        resp_with_tool = _make_anthropic_response(
            text="Let me run that.",
            tool_use_blocks=[{"id": "tu_1", "name": "shell", "input": {"command": "ls"}}],
        )
        # Second call: LLM returns final text
        resp_final = _make_anthropic_response(text="Here are the files.")

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=[resp_with_tool, resp_final])

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            result = await brain.generate([{"role": "user", "content": "list files"}])

        assert result["content"] == "Here are the files."
        assert mock_client.messages.create.call_count == 2
        tool_executor.execute.assert_called_once_with("shell", {"command": "ls"})

    @pytest.mark.asyncio
    async def test_generate_max_tool_rounds_exhausted(self, brain, tool_executor):
        """When max_tool_rounds is reached, stop looping and return with warning."""
        resp_with_tool = _make_anthropic_response(
            text="Calling tool.",
            tool_use_blocks=[{"id": "tu_1", "name": "shell", "input": {"command": "ls"}}],
            input_tokens=5, output_tokens=10,
        )
        mock_client = MagicMock()
        # Always returns tool_calls — never a text-only response
        mock_client.messages.create = AsyncMock(return_value=resp_with_tool)

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            result = await brain.generate(
                [{"role": "user", "content": "loop forever"}],
                max_tool_rounds=1,
            )

        assert "[Max tool rounds reached]" in result["content"]
        # Round 0: call LLM → tool_calls → execute → loop
        # Round 1: call LLM → tool_calls → round_idx >= max_tool_rounds → stop
        assert mock_client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_cumulative_usage(self, brain, tool_executor):
        """Usage tokens should accumulate across multiple rounds."""
        resp_tool = _make_anthropic_response(
            text="",
            tool_use_blocks=[{"id": "tu_1", "name": "shell", "input": {"command": "ls"}}],
            input_tokens=100, output_tokens=50,
        )
        resp_final = _make_anthropic_response(
            text="Done.",
            input_tokens=200, output_tokens=80,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=[resp_tool, resp_final])

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            result = await brain.generate([{"role": "user", "content": "go"}])

        assert result["usage"]["input_tokens"] == 300
        assert result["usage"]["output_tokens"] == 130
        assert result["usage"]["total_tokens"] == 430

    @pytest.mark.asyncio
    async def test_generate_does_not_mutate_caller_messages(self, brain, tool_executor):
        """The caller's message list must not be mutated by the agentic loop."""
        resp_tool = _make_anthropic_response(
            text="",
            tool_use_blocks=[{"id": "tu_1", "name": "shell", "input": {"command": "ls"}}],
        )
        resp_final = _make_anthropic_response(text="Done.")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=[resp_tool, resp_final])

        original_messages = [{"role": "user", "content": "hi"}]
        original_len = len(original_messages)

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            await brain.generate(original_messages)

        assert len(original_messages) == original_len

    @pytest.mark.asyncio
    async def test_generate_tool_results_passed_to_llm(self, brain, tool_executor):
        """Verify that tool_result messages are included when re-calling the LLM."""
        resp_tool = _make_anthropic_response(
            text="Running.",
            tool_use_blocks=[{"id": "tu_1", "name": "shell", "input": {"command": "pwd"}}],
        )
        resp_final = _make_anthropic_response(text="You are in /home.")

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=[resp_tool, resp_final])
        tool_executor.execute = AsyncMock(return_value={"stdout": "/home"})

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            await brain.generate([{"role": "user", "content": "where am I?"}])

        # Second call should have more messages (assistant + tool_result appended)
        second_call_kwargs = mock_client.messages.create.call_args_list[1].kwargs
        second_messages = second_call_kwargs["messages"]
        # Should contain: user, assistant(tool_use blocks), user(tool_result blocks)
        assert len(second_messages) >= 3
        # Last message should be a user message with tool_result content blocks
        last_msg = second_messages[-1]
        assert last_msg["role"] == "user"
        assert isinstance(last_msg["content"], list)
        assert last_msg["content"][0]["type"] == "tool_result"
        assert last_msg["content"][0]["tool_use_id"] == "tu_1"

    @pytest.mark.asyncio
    async def test_generate_multi_round_loop(self, brain, tool_executor):
        """Verify 3-round tool loop: tool → tool → text."""
        resp_tool_1 = _make_anthropic_response(
            text="Step 1",
            tool_use_blocks=[{"id": "tu_1", "name": "shell", "input": {"command": "ls"}}],
            input_tokens=10, output_tokens=5,
        )
        resp_tool_2 = _make_anthropic_response(
            text="Step 2",
            tool_use_blocks=[{"id": "tu_2", "name": "read_file", "input": {"path": "/tmp/x"}}],
            input_tokens=20, output_tokens=10,
        )
        resp_final = _make_anthropic_response(
            text="All done.",
            input_tokens=30, output_tokens=15,
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=[resp_tool_1, resp_tool_2, resp_final])

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            result = await brain.generate([{"role": "user", "content": "do it all"}])

        assert result["content"] == "All done."
        assert mock_client.messages.create.call_count == 3
        assert tool_executor.execute.call_count == 2
        assert result["usage"]["input_tokens"] == 60
        assert result["usage"]["output_tokens"] == 30

    @pytest.mark.asyncio
    async def test_generate_no_tool_executor_skips_loop(self, brain_no_tools):
        """Without tool_executor, tool_calls in response are returned as-is (no loop)."""
        resp = _make_anthropic_response(
            text="I would call a tool.",
            tool_use_blocks=[{"id": "tu_1", "name": "shell", "input": {"command": "ls"}}],
        )
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=resp)

        with patch.object(brain_no_tools, "_get_anthropic_client", return_value=mock_client):
            result = await brain_no_tools.generate([{"role": "user", "content": "hi"}])

        # Should return immediately without executing tools
        assert mock_client.messages.create.call_count == 1
        assert result["content"] == "I would call a tool."

    @pytest.mark.asyncio
    async def test_generate_failover(self, brain):
        """If primary provider fails, should failover to secondary."""
        mock_anthropic = MagicMock()
        mock_anthropic.messages.create = AsyncMock(side_effect=RuntimeError("anthropic down"))

        resp_openai = _make_openai_response(text="Failover response")
        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=resp_openai)

        def get_provider(name):
            if name == "openai":
                return mock_openai
            raise ValueError(f"unknown: {name}")

        with patch.object(brain, "_get_anthropic_client", return_value=mock_anthropic), \
             patch.object(brain, "_get_provider_client", side_effect=get_provider):
            result = await brain.generate([{"role": "user", "content": "hi"}])

        assert result["content"] == "Failover response"

    @pytest.mark.asyncio
    async def test_generate_all_providers_fail(self, brain):
        """If all providers fail, should return error message."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=RuntimeError("all down"))

        mock_openai = MagicMock()
        mock_openai.chat.completions.create = AsyncMock(side_effect=RuntimeError("openai too"))

        def get_provider(name):
            if name == "openai":
                return mock_openai
            raise ValueError(f"unknown: {name}")

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client), \
             patch.object(brain, "_get_provider_client", side_effect=get_provider):
            result = await brain.generate([{"role": "user", "content": "hi"}])

        assert "Error" in result["content"]
        assert result["tool_calls"] == []


# ══════════════════════════════════════════════════════════════
#  PROVIDER CLIENT RESOLUTION
# ══════════════════════════════════════════════════════════════


class TestProviderClient:

    def test_get_provider_anthropic(self, brain):
        with patch.object(brain, "_get_anthropic_client", return_value="mock_anthropic"):
            result = brain._get_provider_client("anthropic")
        assert result == "mock_anthropic"

    def test_get_provider_openai(self, brain):
        with patch.object(brain, "_get_openai_client", return_value="mock_openai"):
            result = brain._get_provider_client("openai")
        assert result == "mock_openai"

    def test_get_provider_unknown_raises(self, brain):
        with pytest.raises(ValueError, match="Unknown provider"):
            brain._get_provider_client("notexist")


# ══════════════════════════════════════════════════════════════
#  STREAMING WITH TOOL_USE SUPPORT
# ══════════════════════════════════════════════════════════════


# ── Mock helpers for Anthropic streaming events ──────────────

class _StreamEvent:
    """Fake Anthropic raw stream event."""
    def __init__(self, event_type, **kwargs):
        self.type = event_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _ContentBlock:
    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _Delta:
    def __init__(self, delta_type, **kwargs):
        self.type = delta_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockMessageStream:
    """Mock for ``client.messages.stream()`` async context manager."""
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for event in self._events:
            yield event


def _make_text_stream_events(text):
    """Build stream events for a text-only Anthropic response."""
    return [
        _StreamEvent("content_block_start", content_block=_ContentBlock("text")),
        _StreamEvent("content_block_delta", delta=_Delta("text_delta", text=text)),
        _StreamEvent("content_block_stop"),
        _StreamEvent("message_stop"),
    ]


def _make_tool_stream_events(text, tool_calls):
    """Build stream events for an Anthropic response with text + tool_use blocks."""
    events = []
    if text:
        events.append(_StreamEvent("content_block_start", content_block=_ContentBlock("text")))
        events.append(_StreamEvent("content_block_delta", delta=_Delta("text_delta", text=text)))
        events.append(_StreamEvent("content_block_stop"))
    for tc in tool_calls:
        events.append(_StreamEvent(
            "content_block_start",
            content_block=_ContentBlock("tool_use", id=tc["id"], name=tc["name"]),
        ))
        json_str = json.dumps(tc.get("arguments", tc.get("input", {})))
        events.append(_StreamEvent(
            "content_block_delta",
            delta=_Delta("input_json_delta", partial_json=json_str),
        ))
        events.append(_StreamEvent("content_block_stop"))
    events.append(_StreamEvent("message_stop"))
    return events


class TestStreamAnthropicEvents:
    """Tests for _stream_anthropic() structured event yielding."""

    @pytest.mark.asyncio
    async def test_text_only(self, brain):
        """Text-only stream yields text events, no tool_calls."""
        events = _make_text_stream_events("Hello world")
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_MockMessageStream(events))

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            collected = []
            async for event in brain._stream_anthropic(
                "model", [{"role": "user", "content": "hi"}], "sys", 0.7, 4096
            ):
                collected.append(event)

        assert len(collected) == 1
        assert collected[0] == {"type": "text", "content": "Hello world"}

    @pytest.mark.asyncio
    async def test_with_tool_calls(self, brain):
        """Stream with tool_use blocks yields text + tool_calls event."""
        events = _make_tool_stream_events("Searching.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}},
        ])
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_MockMessageStream(events))

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            collected = []
            async for event in brain._stream_anthropic(
                "model", [{"role": "user", "content": "ls"}], "sys", 0.7, 4096
            ):
                collected.append(event)

        text_events = [e for e in collected if e["type"] == "text"]
        tool_events = [e for e in collected if e["type"] == "tool_calls"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "Searching."
        assert len(tool_events) == 1
        assert tool_events[0]["tool_calls"][0]["name"] == "shell"
        assert tool_events[0]["tool_calls"][0]["arguments"] == {"command": "ls"}

    @pytest.mark.asyncio
    async def test_partial_json_accumulated(self, brain):
        """Multiple input_json_delta chunks are accumulated correctly."""
        events = [
            _StreamEvent("content_block_start",
                content_block=_ContentBlock("tool_use", id="tu_1", name="shell")),
            _StreamEvent("content_block_delta",
                delta=_Delta("input_json_delta", partial_json='{"com')),
            _StreamEvent("content_block_delta",
                delta=_Delta("input_json_delta", partial_json='mand":')),
            _StreamEvent("content_block_delta",
                delta=_Delta("input_json_delta", partial_json=' "ls -la"}')),
            _StreamEvent("content_block_stop"),
            _StreamEvent("message_stop"),
        ]
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_MockMessageStream(events))

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            collected = []
            async for event in brain._stream_anthropic(
                "model", [{"role": "user", "content": "ls"}], "sys", 0.7, 4096
            ):
                collected.append(event)

        assert len(collected) == 1
        assert collected[0]["type"] == "tool_calls"
        assert collected[0]["tool_calls"][0]["arguments"] == {"command": "ls -la"}

    @pytest.mark.asyncio
    async def test_multiple_tools_in_one_response(self, brain):
        """Two tool_use blocks in one stream are both accumulated."""
        events = _make_tool_stream_events("Checking.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}},
            {"id": "tu_2", "name": "read_file", "arguments": {"path": "/tmp/x"}},
        ])
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_MockMessageStream(events))

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            collected = []
            async for event in brain._stream_anthropic(
                "model", [{"role": "user", "content": "go"}], "sys", 0.7, 4096
            ):
                collected.append(event)

        tool_event = [e for e in collected if e["type"] == "tool_calls"][0]
        assert len(tool_event["tool_calls"]) == 2
        assert tool_event["tool_calls"][0]["name"] == "shell"
        assert tool_event["tool_calls"][1]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_injects_tool_definitions(self, brain):
        """When tool_executor is present, tools kwarg is passed to the stream."""
        events = _make_text_stream_events("ok")
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_MockMessageStream(events))

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            async for _ in brain._stream_anthropic(
                "model", [{"role": "user", "content": "hi"}], "sys", 0.7, 4096
            ):
                pass

        call_kwargs = mock_client.messages.stream.call_args.kwargs
        assert "tools" in call_kwargs
        assert any(t["name"] == "shell" for t in call_kwargs["tools"])


class TestGenerateStreamWithTools:
    """Tests for generate_stream_with_tools() agentic streaming loop."""

    @pytest.mark.asyncio
    async def test_text_only_no_loop(self, brain):
        """When no tool_calls, just yields text and returns."""
        events = _make_text_stream_events("Hello!")
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_MockMessageStream(events))

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            chunks = []
            async for chunk in brain.generate_stream_with_tools(
                [{"role": "user", "content": "hi"}]
            ):
                chunks.append(chunk)

        assert "".join(chunks) == "Hello!"

    @pytest.mark.asyncio
    async def test_executes_tools_and_restreams(self, brain, tool_executor):
        """Tool calls -> execute -> re-stream -> final text."""
        events_r1 = _make_tool_stream_events("Let me check.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}},
        ])
        events_r2 = _make_text_stream_events("Here are the files.")

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(
            side_effect=[_MockMessageStream(events_r1), _MockMessageStream(events_r2)]
        )

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            chunks = []
            async for chunk in brain.generate_stream_with_tools(
                [{"role": "user", "content": "list files"}]
            ):
                chunks.append(chunk)

        full = "".join(chunks)
        assert "Let me check." in full
        assert "\n[TOOL_EXECUTING: shell]\n" in full
        assert "Here are the files." in full
        tool_executor.execute.assert_called_once_with("shell", {"command": "ls"})

    @pytest.mark.asyncio
    async def test_max_tool_rounds_stops(self, brain, tool_executor):
        """Stops at max_tool_rounds with warning marker."""
        tool_events = _make_tool_stream_events("Calling.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}},
        ])
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(
            side_effect=lambda **kw: _MockMessageStream(tool_events)
        )

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            chunks = []
            async for chunk in brain.generate_stream_with_tools(
                [{"role": "user", "content": "loop"}],
                max_tool_rounds=1,
            ):
                chunks.append(chunk)

        full = "".join(chunks)
        assert "[Max tool rounds reached]" in full
        # Round 0: stream + tool call + execute; Round 1: stream + tool call + stop
        assert mock_client.messages.stream.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_results_passed_in_next_round(self, brain, tool_executor):
        """After tool execution, next stream call includes tool_result messages."""
        events_r1 = _make_tool_stream_events("Running.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "pwd"}},
        ])
        events_r2 = _make_text_stream_events("Done.")

        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(
            side_effect=[_MockMessageStream(events_r1), _MockMessageStream(events_r2)]
        )
        tool_executor.execute = AsyncMock(return_value={"stdout": "/home"})

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            async for _ in brain.generate_stream_with_tools(
                [{"role": "user", "content": "where?"}]
            ):
                pass

        # Second stream call should have more messages
        second_kwargs = mock_client.messages.stream.call_args_list[1].kwargs
        second_messages = second_kwargs["messages"]
        # Last message should be a user message with tool_result blocks
        last_msg = second_messages[-1]
        assert last_msg["role"] == "user"
        assert isinstance(last_msg["content"], list)
        assert last_msg["content"][0]["type"] == "tool_result"
        assert last_msg["content"][0]["tool_use_id"] == "tu_1"

    @pytest.mark.asyncio
    async def test_no_executor_skips_tool_loop(self, brain_no_tools):
        """Without tool_executor, tool_calls are ignored (no loop)."""
        events = _make_tool_stream_events("I would use a tool.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}},
        ])
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_MockMessageStream(events))

        with patch.object(brain_no_tools, "_get_anthropic_client", return_value=mock_client):
            chunks = []
            async for chunk in brain_no_tools.generate_stream_with_tools(
                [{"role": "user", "content": "hi"}]
            ):
                chunks.append(chunk)

        assert "".join(chunks) == "I would use a tool."
        assert mock_client.messages.stream.call_count == 1


class TestGenerateStreamBackwardCompat:
    """generate_stream() must still yield plain strings."""

    @pytest.mark.asyncio
    async def test_yields_plain_strings(self, brain):
        events = _make_text_stream_events("Hi there!")
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_MockMessageStream(events))

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            chunks = []
            async for chunk in brain.generate_stream(
                [{"role": "user", "content": "hi"}]
            ):
                chunks.append(chunk)

        assert all(isinstance(c, str) for c in chunks)
        assert "".join(chunks) == "Hi there!"

    @pytest.mark.asyncio
    async def test_ignores_tool_events(self, brain):
        """generate_stream() silently drops tool_calls events."""
        events = _make_tool_stream_events("Text part.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}},
        ])
        mock_client = MagicMock()
        mock_client.messages.stream = MagicMock(return_value=_MockMessageStream(events))

        with patch.object(brain, "_get_anthropic_client", return_value=mock_client):
            chunks = []
            async for chunk in brain.generate_stream(
                [{"role": "user", "content": "hi"}]
            ):
                chunks.append(chunk)

        assert "".join(chunks) == "Text part."
