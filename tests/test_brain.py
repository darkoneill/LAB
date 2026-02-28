"""
Tests for openclaw/agent/brain.py
Mocks ProviderBase instances (not raw SDK clients).
Covers: system message, tool definitions, tool execution routing,
        provider failover, tool_call parsing, CoT extraction, context trimming,
        streaming with tool_use support.
"""

import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

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


# ── Mock provider helpers ────────────────────────────────────


def _make_result(content="Hello!", tool_calls=None, input_tokens=10, output_tokens=20):
    """Build a standard ProviderBase.generate() result dict."""
    return {
        "content": content,
        "model": "mock-model",
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "tool_calls": tool_calls or [],
    }


def _mock_provider(name="anthropic", supports_tools=True):
    """Create a mock ProviderBase with configurable generate()."""
    p = MagicMock()
    type(p).name = PropertyMock(return_value=name)
    type(p).supports_tools = PropertyMock(return_value=supports_tools)
    p.generate = AsyncMock(return_value=_make_result())
    return p


class _StreamMock:
    """Mock for an async generator method (like provider.stream).

    Records calls and returns pre-configured event sequences.
    """
    def __init__(self, *event_sequences):
        self._sequences = list(event_sequences)
        self._call_idx = 0
        self.call_count = 0
        self.call_args_list = []

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.call_args_list.append((args, kwargs))
        events = self._sequences[self._call_idx % len(self._sequences)]
        self._call_idx += 1
        return self._aiter(events)

    @staticmethod
    async def _aiter(events):
        for event in events:
            yield event


def _text_events(text):
    """Build stream events for a text-only response."""
    return [{"type": "text", "content": text}]


def _tool_events(text, tool_calls):
    """Build stream events for a response with text + tool_calls."""
    events = []
    if text:
        events.append({"type": "text", "content": text})
    events.append({"type": "tool_calls", "tool_calls": tool_calls})
    return events


# ── Fixtures ──────────────────────────────────────────────────


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
#  PROVIDER DISPATCH — generate via ProviderBase
# ══════════════════════════════════════════════════════════════


class TestCallProvider:

    @pytest.mark.asyncio
    async def test_anthropic_text_only(self, brain):
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(return_value=_make_result("Hello world"))
        brain._providers["anthropic"] = provider

        result = await brain._call_provider(
            "anthropic", "claude-sonnet-4-20250514",
            [{"role": "user", "content": "hi"}], "sys", 0.7, 4096,
        )
        assert result["content"] == "Hello world"
        assert result["tool_calls"] == []
        assert result["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_anthropic_with_tool_calls(self, brain):
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(return_value=_make_result(
            "Let me search.",
            tool_calls=[
                {"id": "tu_1", "name": "shell", "arguments": {"command": "ls -la"}},
                {"id": "tu_2", "name": "read_file", "arguments": {"path": "/tmp/test"}},
            ],
        ))
        brain._providers["anthropic"] = provider

        result = await brain._call_provider(
            "anthropic", "claude-sonnet-4-20250514",
            [{"role": "user", "content": "list files"}], "sys", 0.7, 4096,
        )
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["name"] == "shell"
        assert result["tool_calls"][0]["arguments"]["command"] == "ls -la"
        assert result["tool_calls"][1]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_anthropic_passes_tools(self, brain):
        """When tool_executor is present, tools are passed to provider."""
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(return_value=_make_result())
        brain._providers["anthropic"] = provider

        await brain._call_provider(
            "anthropic", "model", [{"role": "user", "content": "hi"}], "sys", 0.7, 4096,
        )
        call_kwargs = provider.generate.call_args.kwargs
        assert "tools" in call_kwargs
        assert any(t["name"] == "shell" for t in call_kwargs["tools"])

    @pytest.mark.asyncio
    async def test_anthropic_no_tools_without_executor(self, brain_no_tools):
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(return_value=_make_result())
        brain_no_tools._providers["anthropic"] = provider

        await brain_no_tools._call_provider(
            "anthropic", "model", [{"role": "user", "content": "hi"}], "sys", 0.7, 4096,
        )
        call_kwargs = provider.generate.call_args.kwargs
        assert call_kwargs.get("tools") is None

    @pytest.mark.asyncio
    async def test_openai_text_only(self, brain):
        provider = _mock_provider("openai")
        provider.generate = AsyncMock(return_value=_make_result("Hello!"))
        brain._providers["openai"] = provider

        result = await brain._call_provider(
            "openai", "gpt-4o",
            [{"role": "user", "content": "hi"}], "sys", 0.7, 4096,
        )
        assert result["content"] == "Hello!"
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_openai_with_tool_calls(self, brain):
        provider = _mock_provider("openai")
        provider.generate = AsyncMock(return_value=_make_result(
            "",
            tool_calls=[{"id": "call_1", "name": "shell", "arguments": {"command": "ls"}}],
        ))
        brain._providers["openai"] = provider

        result = await brain._call_provider(
            "openai", "gpt-4o",
            [{"role": "user", "content": "list"}], "sys", 0.7, 4096,
        )
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["arguments"]["command"] == "ls"

    @pytest.mark.asyncio
    async def test_ollama_no_tools(self, brain, fake_settings):
        """Ollama provider (supports_tools=False) should not receive tool definitions."""
        fake_settings._data["providers.ollama.enabled"] = True
        fake_settings._data["providers.ollama.base_url"] = "http://localhost:11434"

        provider = _mock_provider("ollama", supports_tools=False)
        provider.generate = AsyncMock(return_value=_make_result("ok"))
        brain._providers["ollama"] = provider

        await brain._call_provider(
            "ollama", "llama3",
            [{"role": "user", "content": "hi"}], "sys", 0.7, 4096,
        )
        call_kwargs = provider.generate.call_args.kwargs
        assert call_kwargs.get("tools") is None

    @pytest.mark.asyncio
    async def test_openai_passes_tools(self, brain):
        provider = _mock_provider("openai")
        provider.generate = AsyncMock(return_value=_make_result("ok"))
        brain._providers["openai"] = provider

        await brain._call_provider(
            "openai", "gpt-4o",
            [{"role": "user", "content": "hi"}], "sys", 0.7, 4096,
        )
        call_kwargs = provider.generate.call_args.kwargs
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
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(return_value=_make_result("Hello user!"))
        brain._providers["anthropic"] = provider

        result = await brain.generate([{"role": "user", "content": "hi"}])
        assert result["content"] == "Hello user!"
        assert result["usage"]["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_generate_with_thinking(self, brain):
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(return_value=_make_result(
            "<thinking>This is safe.</thinking>Result here."
        ))
        brain._providers["anthropic"] = provider

        result = await brain.generate([{"role": "user", "content": "do something"}])
        assert result["content"] == "Result here."
        assert result["thinking"] == "This is safe."

    @pytest.mark.asyncio
    async def test_generate_tool_call_loop(self, brain, tool_executor):
        """When LLM returns tool_calls, brain should execute then re-call LLM."""
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(side_effect=[
            _make_result(
                "Let me run that.",
                tool_calls=[{"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}}],
            ),
            _make_result("Here are the files."),
        ])
        brain._providers["anthropic"] = provider

        result = await brain.generate([{"role": "user", "content": "list files"}])
        assert result["content"] == "Here are the files."
        assert provider.generate.call_count == 2
        tool_executor.execute.assert_called_once_with("shell", {"command": "ls"})

    @pytest.mark.asyncio
    async def test_generate_max_tool_rounds_exhausted(self, brain, tool_executor):
        """When max_tool_rounds is reached, stop looping and return with warning."""
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(return_value=_make_result(
            "Calling tool.",
            tool_calls=[{"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}}],
            input_tokens=5, output_tokens=10,
        ))
        brain._providers["anthropic"] = provider

        result = await brain.generate(
            [{"role": "user", "content": "loop forever"}],
            max_tool_rounds=1,
        )
        assert "[Max tool rounds reached]" in result["content"]
        assert provider.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_cumulative_usage(self, brain, tool_executor):
        """Usage tokens should accumulate across multiple rounds."""
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(side_effect=[
            _make_result(
                "",
                tool_calls=[{"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}}],
                input_tokens=100, output_tokens=50,
            ),
            _make_result("Done.", input_tokens=200, output_tokens=80),
        ])
        brain._providers["anthropic"] = provider

        result = await brain.generate([{"role": "user", "content": "go"}])
        assert result["usage"]["input_tokens"] == 300
        assert result["usage"]["output_tokens"] == 130
        assert result["usage"]["total_tokens"] == 430

    @pytest.mark.asyncio
    async def test_generate_does_not_mutate_caller_messages(self, brain, tool_executor):
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(side_effect=[
            _make_result(
                "",
                tool_calls=[{"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}}],
            ),
            _make_result("Done."),
        ])
        brain._providers["anthropic"] = provider

        original_messages = [{"role": "user", "content": "hi"}]
        original_len = len(original_messages)

        await brain.generate(original_messages)
        assert len(original_messages) == original_len

    @pytest.mark.asyncio
    async def test_generate_tool_results_passed_to_llm(self, brain, tool_executor):
        """Verify that tool_result messages are included when re-calling the LLM."""
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(side_effect=[
            _make_result(
                "Running.",
                tool_calls=[{"id": "tu_1", "name": "shell", "arguments": {"command": "pwd"}}],
            ),
            _make_result("You are in /home."),
        ])
        brain._providers["anthropic"] = provider
        tool_executor.execute = AsyncMock(return_value={"stdout": "/home"})

        await brain.generate([{"role": "user", "content": "where am I?"}])

        # Second call should have more messages (assistant + tool_result appended)
        second_call_args = provider.generate.call_args_list[1]
        second_messages = second_call_args[0][1]  # positional arg: messages
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
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(side_effect=[
            _make_result(
                "Step 1",
                tool_calls=[{"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}}],
                input_tokens=10, output_tokens=5,
            ),
            _make_result(
                "Step 2",
                tool_calls=[{"id": "tu_2", "name": "read_file", "arguments": {"path": "/tmp/x"}}],
                input_tokens=20, output_tokens=10,
            ),
            _make_result("All done.", input_tokens=30, output_tokens=15),
        ])
        brain._providers["anthropic"] = provider

        result = await brain.generate([{"role": "user", "content": "do it all"}])
        assert result["content"] == "All done."
        assert provider.generate.call_count == 3
        assert tool_executor.execute.call_count == 2
        assert result["usage"]["input_tokens"] == 60
        assert result["usage"]["output_tokens"] == 30

    @pytest.mark.asyncio
    async def test_generate_no_tool_executor_skips_loop(self, brain_no_tools):
        """Without tool_executor, tool_calls in response are returned as-is (no loop)."""
        provider = _mock_provider("anthropic")
        provider.generate = AsyncMock(return_value=_make_result(
            "I would call a tool.",
            tool_calls=[{"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}}],
        ))
        brain_no_tools._providers["anthropic"] = provider

        result = await brain_no_tools.generate([{"role": "user", "content": "hi"}])
        assert provider.generate.call_count == 1
        assert result["content"] == "I would call a tool."

    @pytest.mark.asyncio
    async def test_generate_failover(self, brain):
        """If primary provider fails, should failover to secondary."""
        provider_anthropic = _mock_provider("anthropic")
        provider_anthropic.generate = AsyncMock(side_effect=RuntimeError("anthropic down"))
        brain._providers["anthropic"] = provider_anthropic

        provider_openai = _mock_provider("openai")
        provider_openai.generate = AsyncMock(return_value=_make_result("Failover response"))
        brain._providers["openai"] = provider_openai

        result = await brain.generate([{"role": "user", "content": "hi"}])
        assert result["content"] == "Failover response"

    @pytest.mark.asyncio
    async def test_generate_all_providers_fail(self, brain):
        """If all providers fail, should return error message."""
        provider_anthropic = _mock_provider("anthropic")
        provider_anthropic.generate = AsyncMock(side_effect=RuntimeError("all down"))
        brain._providers["anthropic"] = provider_anthropic

        provider_openai = _mock_provider("openai")
        provider_openai.generate = AsyncMock(side_effect=RuntimeError("openai too"))
        brain._providers["openai"] = provider_openai

        result = await brain.generate([{"role": "user", "content": "hi"}])
        assert "Error" in result["content"]
        assert result["tool_calls"] == []


# ══════════════════════════════════════════════════════════════
#  STREAMING WITH TOOL_USE SUPPORT (via ProviderBase.stream)
# ══════════════════════════════════════════════════════════════


class TestStreamProvider:
    """Tests for _stream_provider() structured event yielding."""

    @pytest.mark.asyncio
    async def test_text_only(self, brain):
        """Text-only stream yields text events, no tool_calls."""
        provider = _mock_provider("anthropic")
        provider.stream = _StreamMock(_text_events("Hello world"))
        brain._providers["anthropic"] = provider

        collected = []
        async for event in brain._stream_provider(
            "anthropic", "model", [{"role": "user", "content": "hi"}], "sys", 0.7, 4096
        ):
            collected.append(event)

        assert len(collected) == 1
        assert collected[0] == {"type": "text", "content": "Hello world"}

    @pytest.mark.asyncio
    async def test_with_tool_calls(self, brain):
        """Stream with tool_use blocks yields text + tool_calls event."""
        events = _tool_events("Searching.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}},
        ])
        provider = _mock_provider("anthropic")
        provider.stream = _StreamMock(events)
        brain._providers["anthropic"] = provider

        collected = []
        async for event in brain._stream_provider(
            "anthropic", "model", [{"role": "user", "content": "ls"}], "sys", 0.7, 4096
        ):
            collected.append(event)

        text_events = [e for e in collected if e["type"] == "text"]
        tool_call_events = [e for e in collected if e["type"] == "tool_calls"]
        assert len(text_events) == 1
        assert text_events[0]["content"] == "Searching."
        assert len(tool_call_events) == 1
        assert tool_call_events[0]["tool_calls"][0]["name"] == "shell"
        assert tool_call_events[0]["tool_calls"][0]["arguments"] == {"command": "ls"}

    @pytest.mark.asyncio
    async def test_multiple_tools_in_one_response(self, brain):
        """Two tool_use blocks in one stream are both returned."""
        events = _tool_events("Checking.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}},
            {"id": "tu_2", "name": "read_file", "arguments": {"path": "/tmp/x"}},
        ])
        provider = _mock_provider("anthropic")
        provider.stream = _StreamMock(events)
        brain._providers["anthropic"] = provider

        collected = []
        async for event in brain._stream_provider(
            "anthropic", "model", [{"role": "user", "content": "go"}], "sys", 0.7, 4096
        ):
            collected.append(event)

        tool_event = [e for e in collected if e["type"] == "tool_calls"][0]
        assert len(tool_event["tool_calls"]) == 2
        assert tool_event["tool_calls"][0]["name"] == "shell"
        assert tool_event["tool_calls"][1]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_passes_tools_kwarg(self, brain):
        """When tool_executor is present, tools kwarg is passed to stream."""
        provider = _mock_provider("anthropic")
        provider.stream = _StreamMock(_text_events("ok"))
        brain._providers["anthropic"] = provider

        async for _ in brain._stream_provider(
            "anthropic", "model", [{"role": "user", "content": "hi"}], "sys", 0.7, 4096
        ):
            pass

        _args, call_kwargs = provider.stream.call_args_list[0]
        assert "tools" in call_kwargs
        assert any(t["name"] == "shell" for t in call_kwargs["tools"])


class TestGenerateStreamWithTools:
    """Tests for generate_stream_with_tools() agentic streaming loop."""

    @pytest.mark.asyncio
    async def test_text_only_no_loop(self, brain):
        """When no tool_calls, just yields text and returns."""
        provider = _mock_provider("anthropic")
        provider.stream = _StreamMock(_text_events("Hello!"))
        brain._providers["anthropic"] = provider

        chunks = []
        async for chunk in brain.generate_stream_with_tools(
            [{"role": "user", "content": "hi"}]
        ):
            chunks.append(chunk)

        assert "".join(chunks) == "Hello!"

    @pytest.mark.asyncio
    async def test_executes_tools_and_restreams(self, brain, tool_executor):
        """Tool calls -> execute -> re-stream -> final text."""
        events_r1 = _tool_events("Let me check.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}},
        ])
        events_r2 = _text_events("Here are the files.")

        provider = _mock_provider("anthropic")
        provider.stream = _StreamMock(events_r1, events_r2)
        brain._providers["anthropic"] = provider

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
        tool_events_data = _tool_events("Calling.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}},
        ])

        provider = _mock_provider("anthropic")
        provider.stream = _StreamMock(tool_events_data, tool_events_data)
        brain._providers["anthropic"] = provider

        chunks = []
        async for chunk in brain.generate_stream_with_tools(
            [{"role": "user", "content": "loop"}],
            max_tool_rounds=1,
        ):
            chunks.append(chunk)

        full = "".join(chunks)
        assert "[Max tool rounds reached]" in full
        assert provider.stream.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_results_passed_in_next_round(self, brain, tool_executor):
        """After tool execution, next stream call includes tool_result messages."""
        events_r1 = _tool_events("Running.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "pwd"}},
        ])
        events_r2 = _text_events("Done.")

        provider = _mock_provider("anthropic")
        provider.stream = _StreamMock(events_r1, events_r2)
        brain._providers["anthropic"] = provider
        tool_executor.execute = AsyncMock(return_value={"stdout": "/home"})

        async for _ in brain.generate_stream_with_tools(
            [{"role": "user", "content": "where?"}]
        ):
            pass

        # Second stream call should have more messages
        second_args, second_kwargs = provider.stream.call_args_list[1]
        # provider.stream(model_id, formatted, system, tools=..., ...)
        second_messages = second_args[1]  # formatted messages
        # Last message should be a user message with tool_result blocks
        last_msg = second_messages[-1]
        assert last_msg["role"] == "user"
        assert isinstance(last_msg["content"], list)
        assert last_msg["content"][0]["type"] == "tool_result"
        assert last_msg["content"][0]["tool_use_id"] == "tu_1"

    @pytest.mark.asyncio
    async def test_no_executor_skips_tool_loop(self, brain_no_tools):
        """Without tool_executor, tool_calls are ignored (no loop)."""
        events = _tool_events("I would use a tool.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}},
        ])
        provider = _mock_provider("anthropic")
        provider.stream = _StreamMock(events)
        brain_no_tools._providers["anthropic"] = provider

        chunks = []
        async for chunk in brain_no_tools.generate_stream_with_tools(
            [{"role": "user", "content": "hi"}]
        ):
            chunks.append(chunk)

        assert "".join(chunks) == "I would use a tool."
        assert provider.stream.call_count == 1


class TestGenerateStreamBackwardCompat:
    """generate_stream() must still yield plain strings."""

    @pytest.mark.asyncio
    async def test_yields_plain_strings(self, brain):
        provider = _mock_provider("anthropic")
        provider.stream = _StreamMock(_text_events("Hi there!"))
        brain._providers["anthropic"] = provider

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
        events = _tool_events("Text part.", [
            {"id": "tu_1", "name": "shell", "arguments": {"command": "ls"}},
        ])
        provider = _mock_provider("anthropic")
        provider.stream = _StreamMock(events)
        brain._providers["anthropic"] = provider

        chunks = []
        async for chunk in brain.generate_stream(
            [{"role": "user", "content": "hi"}]
        ):
            chunks.append(chunk)

        assert "".join(chunks) == "Text part."
