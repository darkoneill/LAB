"""
Agent Brain - The reasoning engine.
Multi-model, prompt-driven LLM layer inspired by AgentZero's architecture.
Enhanced with Chain of Thought (CoT) reasoning for improved reliability.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional, AsyncGenerator

from openclaw.config.settings import get_settings
from openclaw.gateway.router import RequestRouter
from openclaw.providers.base import ProviderBase
from openclaw.providers.factory import create_provider

logger = logging.getLogger("openclaw.agent.brain")

# Chain of Thought prompt injection
COT_INSTRUCTION = """
## Reasoning Protocol

Before taking any action or responding, you MUST think through the problem step by step.
Use <thinking>...</thinking> tags to show your reasoning process. This is mandatory.

Structure your thinking:
1. **Understand**: What is the user asking? What is the context?
2. **Analyze**: What information or tools do I need? What are the constraints?
3. **Plan**: What steps should I take? In what order?
4. **Validate**: Is my plan safe? Are there any risks?

After your thinking, provide your response or take action.

Example:
<thinking>
The user wants to list files in a directory.
I need to use the shell tool with 'ls -la'.
This is a safe, read-only operation.
</thinking>
[Then execute the action or respond]

IMPORTANT: Always show your thinking process. Never skip the <thinking> tags when:
- Executing shell commands
- Modifying files
- Making decisions that affect system state
"""


class AgentBrain:
    """
    Central reasoning engine that:
    - Interprets user intent
    - Routes to appropriate tools/skills
    - Manages conversation context
    - Supports streaming and multi-model
    - Handles tool calling natively

    Accepts a ``provider`` (ProviderBase) via dependency injection.
    When no provider is given the brain falls back to lazy creation
    via the provider factory + settings.
    """

    def __init__(self, tool_executor=None, skill_router=None, provider: ProviderBase | None = None):
        self.settings = get_settings()
        self.router = RequestRouter()
        self.tool_executor = tool_executor
        self.skill_router = skill_router
        self._provider = provider
        self._providers: dict[str, ProviderBase] = {}
        if provider is not None:
            self._providers[provider.name] = provider
        self._system_prompt = ""
        self._personality = ""
        self._tools_prompt = ""
        self._load_prompts()
        self._tool_definitions = None  # cached tool defs

    # ── Prompt loading ──────────────────────────────────────

    def _load_prompts(self):
        """Load prompt templates from files."""
        base = Path(self.settings._base_dir) if self.settings._base_dir else Path(__file__).parent.parent

        system_file = base / self.settings.get("agent.system_prompt_file", "agent/prompts/system.md")
        personality_file = base / self.settings.get("agent.personality_file", "agent/prompts/personality.md")
        tools_file = base / self.settings.get("agent.tools_prompt_file", "agent/prompts/tools.md")

        for attr, filepath in [
            ("_system_prompt", system_file),
            ("_personality", personality_file),
            ("_tools_prompt", tools_file),
        ]:
            if filepath.exists():
                setattr(self, attr, filepath.read_text(encoding="utf-8"))
            else:
                logger.debug(f"Prompt file not found: {filepath}")

    def _build_system_message(self, memory_context: str = "") -> str:
        """Assemble the full system message from components."""
        parts = [self._system_prompt]
        if self._personality:
            parts.append(f"\n\n## Personality\n{self._personality}")
        if self._tools_prompt:
            parts.append(f"\n\n## Available Tools\n{self._tools_prompt}")
        if memory_context:
            parts.append(f"\n\n## Relevant Memories\n{memory_context}")

        # Add available skills
        if self.skill_router:
            skills_info = self.skill_router.get_skills_description()
            if skills_info:
                parts.append(f"\n\n## Available Skills\n{skills_info}")

        # Add Chain of Thought instruction
        if self.settings.get("agent.chain_of_thought", True):
            parts.append(COT_INSTRUCTION)

        return "\n".join(parts)

    # ── Tool definitions ────────────────────────────────────

    def _get_tool_definitions_anthropic(self) -> list[dict]:
        """Build Anthropic-format tool definitions from executor + skills."""
        if self._tool_definitions is not None:
            return self._tool_definitions

        tools = []

        # Default tools from executor
        if self.tool_executor:
            tools.append({
                "name": "shell",
                "description": "Execute a shell command on the system.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The shell command to execute"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (optional)"},
                    },
                    "required": ["command"],
                },
            })
            tools.append({
                "name": "read_file",
                "description": "Read the contents of a file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file to read"},
                    },
                    "required": ["path"],
                },
            })
            tools.append({
                "name": "write_file",
                "description": "Write content to a file (creates or overwrites).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"},
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["path", "content"],
                },
            })
            tools.append({
                "name": "search_files",
                "description": "Search for files matching a glob pattern.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Root directory to search in"},
                        "pattern": {"type": "string", "description": "Glob pattern to match"},
                    },
                    "required": ["pattern"],
                },
            })

        # Skills as tools
        if self.skill_router:
            for skill_info in self.skill_router.list_skills():
                name = skill_info["name"]
                desc = skill_info.get("description", name)
                # Generic skill invocation schema
                tools.append({
                    "name": f"skill_{name}",
                    "description": f"Skill: {desc}",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The query or input for this skill"},
                        },
                        "required": ["query"],
                    },
                })

        self._tool_definitions = tools
        return tools

    def invalidate_tool_cache(self):
        """Call this when skills are added/removed at runtime."""
        self._tool_definitions = None

    def _get_tool_definitions_openai(self) -> list[dict]:
        """Build OpenAI-format tool definitions (function calling)."""
        anthropic_tools = self._get_tool_definitions_anthropic()
        openai_tools = []
        for t in anthropic_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            })
        return openai_tools

    # ── CoT extraction ──────────────────────────────────────

    def _extract_thinking(self, content: str) -> tuple[str, str]:
        """Extract thinking blocks from response content.

        Returns:
            tuple: (thinking_content, response_without_thinking)
        """
        thinking_pattern = r'<thinking>(.*?)</thinking>'
        thinking_matches = re.findall(thinking_pattern, content, re.DOTALL)
        thinking = "\n".join(thinking_matches).strip()

        # Remove thinking blocks from visible response
        clean_response = re.sub(thinking_pattern, '', content, flags=re.DOTALL).strip()

        return thinking, clean_response

    def _validate_thinking(self, thinking: str, action_type: str = None) -> bool:
        """Validate that thinking contains required safety checks for dangerous actions."""
        if not thinking:
            return False

        thinking_lower = thinking.lower()

        # For shell commands, ensure safety was considered
        if action_type == "shell":
            safety_keywords = ["safe", "risk", "danger", "read-only", "harmless", "verify", "check"]
            return any(kw in thinking_lower for kw in safety_keywords)

        return True

    # ── Provider resolution ─────────────────────────────────

    def _get_provider(self, provider_name: str) -> ProviderBase:
        """Return a ProviderBase for *provider_name*, creating lazily if needed."""
        if provider_name in self._providers:
            return self._providers[provider_name]
        provider = create_provider(provider_name, self.settings)
        self._providers[provider_name] = provider
        return provider

    # Keep legacy helpers for backward compatibility with tests that
    # patch ``_get_anthropic_client`` / ``_get_provider_client``.

    def _get_provider_client(self, provider_name: str):
        """Legacy: return raw SDK client. Prefer _get_provider()."""
        if provider_name == "anthropic":
            return self._get_anthropic_client()
        elif provider_name == "openai":
            return self._get_openai_client()
        elif provider_name == "ollama":
            return self._get_ollama_client()
        elif provider_name == "custom":
            return self._get_custom_client()
        raise ValueError(f"Unknown provider: {provider_name}")

    def _get_anthropic_client(self):
        try:
            import anthropic
            api_key = self.settings.get("providers.anthropic.api_key", "")
            if not api_key:
                import os
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            return anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    def _get_openai_client(self):
        try:
            import openai
            api_key = self.settings.get("providers.openai.api_key", "")
            if not api_key:
                import os
                api_key = os.environ.get("OPENAI_API_KEY", "")
            base_url = self.settings.get("providers.openai.base_url", None) or None
            return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

    def _get_ollama_client(self):
        try:
            import openai
            base_url = self.settings.get("providers.ollama.base_url", "http://localhost:11434")
            return openai.AsyncOpenAI(api_key="ollama", base_url=f"{base_url}/v1")
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

    def _get_custom_client(self):
        try:
            import openai
            base_url = self.settings.get("providers.custom.base_url", "")
            api_key = self.settings.get("providers.custom.api_key", "")
            return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")

    # ── Message formatting ──────────────────────────────────

    def _format_messages_anthropic(self, messages: list[dict], system_msg: str) -> tuple[str, list[dict]]:
        """Format messages for Anthropic API.

        Handles special internal roles:
        - assistant messages with 'tool_calls' → content blocks (text + tool_use)
        - 'tool_result' messages → user message with tool_result content blocks
        """
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                continue

            # Assistant message that used tools → structured content blocks
            if role == "assistant" and msg.get("tool_calls"):
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in msg["tool_calls"]:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc.get("arguments", {}),
                    })
                formatted.append({"role": "assistant", "content": blocks})
                continue

            # Tool results → user message with tool_result content blocks
            if role == "tool_result":
                blocks = []
                for tr in msg.get("tool_results", []):
                    blocks.append({
                        "type": "tool_result",
                        "tool_use_id": tr["tool_use_id"],
                        "content": tr.get("content", ""),
                    })
                formatted.append({"role": "user", "content": blocks})
                continue

            if role not in ("user", "assistant"):
                role = "user"
            # Avoid consecutive same-role messages (only for plain text)
            if formatted and formatted[-1]["role"] == role and isinstance(formatted[-1].get("content"), str):
                formatted[-1]["content"] += f"\n{content}"
            else:
                formatted.append({"role": role, "content": content})

        # Ensure first message is from user
        if not formatted or formatted[0]["role"] != "user":
            formatted.insert(0, {"role": "user", "content": "(conversation start)"})

        # Context window trimming: ~100k tokens safety limit (4 chars ≈ 1 token)
        max_context_chars = 400_000
        total_chars = sum(len(str(m.get("content", ""))) for m in formatted)
        while total_chars > max_context_chars and len(formatted) > 1:
            removed = formatted.pop(0)
            total_chars -= len(str(removed.get("content", "")))

        return system_msg, formatted

    def _format_messages_openai(self, messages: list[dict], system_msg: str) -> list[dict]:
        """Format messages for OpenAI-compatible API.

        Handles special internal roles:
        - assistant messages with 'tool_calls' → assistant msg with tool_calls array
        - 'tool_result' messages → one 'tool' message per result
        """
        formatted = [{"role": "system", "content": system_msg}]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Assistant message that used tools
            if role == "assistant" and msg.get("tool_calls"):
                oai_tool_calls = []
                for tc in msg["tool_calls"]:
                    oai_tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                    })
                formatted.append({
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": oai_tool_calls,
                })
                continue

            # Tool results → one tool message per result
            if role == "tool_result":
                for tr in msg.get("tool_results", []):
                    formatted.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": tr.get("content", ""),
                    })
                continue

            if role not in ("system", "user", "assistant"):
                role = "user"
            formatted.append({"role": role, "content": content})
        return formatted

    # ── Core generation ─────────────────────────────────────

    async def generate(
        self,
        messages: list[dict],
        memory_context: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_tool_rounds: int = 10,
    ) -> dict:
        """Generate a response with an agentic tool-calling loop.

        If the LLM returns tool_calls, they are executed and the results are
        fed back as properly formatted tool_result messages (Anthropic format
        or OpenAI tool role).  The loop continues until the LLM produces a
        text-only response or ``max_tool_rounds`` is exhausted.

        Args:
            messages: Conversation history (internal format).
            memory_context: Injected memory text for the system prompt.
            model: Explicit model override (provider/model or model id).
            temperature: Sampling temperature override.
            max_tokens: Max output tokens override.
            max_tool_rounds: Maximum tool-call round-trips before stopping.
        """
        temp = temperature or self.settings.get("agent.temperature", 0.7)
        max_tok = max_tokens or self.settings.get("agent.max_tokens", 4096)
        system_msg = self._build_system_message(memory_context)

        provider_name, model_id = self.router.resolve_model(model)
        # Accumulate token usage across rounds
        cumulative_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # Work on a copy so the caller's list is not mutated
        loop_messages = list(messages)

        for round_idx in range(max_tool_rounds + 1):
            try:
                start_time = time.time()
                result = await self._call_provider(provider_name, model_id, loop_messages, system_msg, temp, max_tok)
                latency = (time.time() - start_time) * 1000
                self.router.record_success(provider_name, latency, result.get("usage", {}).get("total_tokens", 0))

                # Accumulate usage
                for k in cumulative_usage:
                    cumulative_usage[k] += result.get("usage", {}).get(k, 0)

            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                self.router.record_failure(provider_name)

                # Try failover
                failover = self.router.get_failover(provider_name, model_id)
                if failover:
                    fo_provider, fo_model = failover
                    try:
                        start_time = time.time()
                        result = await self._call_provider(fo_provider, fo_model, loop_messages, system_msg, temp, max_tok)
                        latency = (time.time() - start_time) * 1000
                        self.router.record_success(fo_provider, latency)
                        for k in cumulative_usage:
                            cumulative_usage[k] += result.get("usage", {}).get(k, 0)
                    except Exception as e2:
                        logger.error(f"Failover to {fo_provider} also failed: {e2}")
                        return {"content": f"Error: {str(e)}", "model": model_id, "usage": cumulative_usage, "tool_calls": []}
                else:
                    return {"content": f"Error: {str(e)}", "model": model_id, "usage": cumulative_usage, "tool_calls": []}

            # Extract and log thinking
            if self.settings.get("agent.chain_of_thought", True):
                thinking, clean_content = self._extract_thinking(result["content"])
                if thinking:
                    logger.debug(f"Agent thinking: {thinking[:200]}...")
                    result["thinking"] = thinking
                    result["content"] = clean_content

            # ── No tool calls → final answer ──
            if not result.get("tool_calls") or not self.tool_executor:
                result["usage"] = cumulative_usage
                return result

            # ── Tool calls → execute and loop ──
            if round_idx >= max_tool_rounds:
                logger.warning(f"Max tool rounds ({max_tool_rounds}) reached")
                result["usage"] = cumulative_usage
                result["content"] = result.get("content", "") + "\n[Max tool rounds reached]"
                return result

            tool_calls = result["tool_calls"]
            tool_results = await self._execute_tools(tool_calls)

            # Append assistant message (with tool_calls metadata for formatting)
            loop_messages.append({
                "role": "assistant",
                "content": result.get("content", ""),
                "tool_calls": tool_calls,
            })

            # Append tool_result message (proper Anthropic / OpenAI format)
            tool_result_entries = []
            for tc, tr in zip(tool_calls, tool_results):
                content_str = json.dumps(tr.get("result", tr.get("error", "")), default=str)
                tool_result_entries.append({
                    "tool_use_id": tc["id"],
                    "content": content_str,
                })
            loop_messages.append({
                "role": "tool_result",
                "tool_results": tool_result_entries,
            })

        # Should not reach here, but safety net
        return {"content": "[Max tool rounds reached]", "model": model_id, "usage": cumulative_usage, "tool_calls": []}

    async def generate_stream(
        self,
        messages: list[dict],
        memory_context: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens (text only, no tool execution)."""
        temp = temperature or self.settings.get("agent.temperature", 0.7)
        max_tok = max_tokens or self.settings.get("agent.max_tokens", 4096)
        system_msg = self._build_system_message(memory_context)
        provider_name, model_id = self.router.resolve_model(model)

        try:
            if provider_name == "anthropic":
                async for event in self._stream_anthropic(model_id, messages, system_msg, temp, max_tok):
                    if isinstance(event, dict) and event.get("type") == "text":
                        yield event["content"]
                    # tool_calls events are silently dropped in text-only mode
            else:
                async for chunk in self._stream_openai(provider_name, model_id, messages, system_msg, temp, max_tok):
                    yield chunk
        except Exception as e:
            logger.error(f"Streaming error with {provider_name}: {e}")
            yield f"\n[Error: {str(e)}]"

    async def generate_stream_with_tools(
        self,
        messages: list[dict],
        memory_context: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_tool_rounds: int = 10,
    ) -> AsyncGenerator[str, None]:
        """Stream response with agentic tool execution.

        Yields text tokens in real time.  When the model returns tool_use
        blocks the method yields ``\\n[TOOL_EXECUTING: <name>]\\n`` markers,
        executes the tools, injects the results into the conversation, and
        starts a new stream.  Loops until the model produces a text-only
        response or *max_tool_rounds* is reached.
        """
        temp = temperature or self.settings.get("agent.temperature", 0.7)
        max_tok = max_tokens or self.settings.get("agent.max_tokens", 4096)
        system_msg = self._build_system_message(memory_context)
        provider_name, model_id = self.router.resolve_model(model)

        loop_messages = list(messages)

        for round_idx in range(max_tool_rounds + 1):
            accumulated_text = ""
            tool_calls: list[dict] = []

            try:
                if provider_name == "anthropic":
                    async for event in self._stream_anthropic(
                        model_id, loop_messages, system_msg, temp, max_tok
                    ):
                        if event.get("type") == "text":
                            accumulated_text += event["content"]
                            yield event["content"]
                        elif event.get("type") == "tool_calls":
                            tool_calls = event["tool_calls"]
                else:
                    # Non-Anthropic providers: text-only stream, no tool loop
                    async for chunk in self._stream_openai(
                        provider_name, model_id, loop_messages, system_msg, temp, max_tok
                    ):
                        yield chunk
                    return
            except Exception as e:
                logger.error(f"Stream tool error on round {round_idx}: {e}")
                yield f"\n[Error: {str(e)}]"
                return

            # No tool calls or no executor → done
            if not tool_calls or not self.tool_executor:
                return

            if round_idx >= max_tool_rounds:
                yield "\n[Max tool rounds reached]"
                return

            # Signal tool execution to the client
            for tc in tool_calls:
                yield f"\n[TOOL_EXECUTING: {tc['name']}]\n"

            # Execute tools
            tool_results = await self._execute_tools(tool_calls)

            # Append assistant + tool_result messages for the next round
            loop_messages.append({
                "role": "assistant",
                "content": accumulated_text,
                "tool_calls": tool_calls,
            })

            tool_result_entries = []
            for tc, tr in zip(tool_calls, tool_results):
                content_str = json.dumps(
                    tr.get("result", tr.get("error", "")), default=str
                )
                tool_result_entries.append({
                    "tool_use_id": tc["id"],
                    "content": content_str,
                })
            loop_messages.append({
                "role": "tool_result",
                "tool_results": tool_result_entries,
            })

    # ── Provider dispatch (uses ProviderBase when available) ─

    async def _call_provider(
        self, provider_name: str, model_id: str, messages: list[dict],
        system_msg: str, temperature: float, max_tokens: int
    ) -> dict:
        """Call a specific provider.

        Delegates to a ProviderBase instance when one is registered,
        otherwise falls back to the legacy direct-client code path
        (kept for backward compatibility with existing tests).
        """
        if provider_name in self._providers:
            provider = self._providers[provider_name]
            if provider_name == "anthropic":
                system, formatted = self._format_messages_anthropic(messages, system_msg)
                tools = self._get_tool_definitions_anthropic() if self.tool_executor else None
                return await provider.generate(model_id, formatted, system, tools=tools, temperature=temperature, max_tokens=max_tokens)
            else:
                formatted = self._format_messages_openai(messages, system_msg)
                tools = self._get_tool_definitions_openai() if self.tool_executor and provider.supports_tools else None
                # OpenAIProvider.generate expects messages WITHOUT a prepended system message
                # (it prepends one itself). Strip it to avoid duplication.
                raw_formatted = formatted[1:] if formatted and formatted[0].get("role") == "system" else formatted
                return await provider.generate(model_id, raw_formatted, system_msg, tools=tools, temperature=temperature, max_tokens=max_tokens)

        # Legacy path — direct SDK clients (used by existing tests that patch _get_anthropic_client)
        if provider_name == "anthropic":
            return await self._call_anthropic(model_id, messages, system_msg, temperature, max_tokens)
        else:
            return await self._call_openai_compat(provider_name, model_id, messages, system_msg, temperature, max_tokens)

    async def _call_anthropic(
        self, model_id: str, messages: list[dict], system_msg: str,
        temperature: float, max_tokens: int
    ) -> dict:
        client = self._get_anthropic_client()
        system, formatted = self._format_messages_anthropic(messages, system_msg)

        # Build request kwargs
        kwargs = dict(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=formatted,
        )

        # Inject native tool definitions if executor is available
        tools = self._get_tool_definitions_anthropic()
        if tools and self.tool_executor:
            kwargs["tools"] = tools

        response = await client.messages.create(**kwargs)

        content = ""
        tool_calls = []
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input if isinstance(block.input, dict) else {},
                })

        return {
            "content": content,
            "model": model_id,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            "tool_calls": tool_calls,
        }

    async def _call_openai_compat(
        self, provider_name: str, model_id: str, messages: list[dict],
        system_msg: str, temperature: float, max_tokens: int
    ) -> dict:
        client = self._get_provider_client(provider_name)
        formatted = self._format_messages_openai(messages, system_msg)

        kwargs = dict(
            model=model_id,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Inject native tool definitions (skip for ollama which may not support it)
        if provider_name not in ("ollama",) and self.tool_executor:
            tools = self._get_tool_definitions_openai()
            if tools:
                kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        usage = response.usage

        # Parse tool calls from OpenAI response
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {"query": args}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                })

        return {
            "content": choice.message.content or "",
            "model": model_id,
            "usage": {
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "tool_calls": tool_calls,
        }

    async def _stream_anthropic(
        self, model_id: str, messages: list[dict], system_msg: str,
        temperature: float, max_tokens: int
    ) -> AsyncGenerator[dict, None]:
        """Stream Anthropic response, yielding structured events.

        Yields:
            ``{"type": "text", "content": str}``  for text deltas
            ``{"type": "tool_calls", "tool_calls": list[dict]}``  at message_stop when
            tool_use blocks were accumulated during the stream.
        """
        client = self._get_anthropic_client()
        system, formatted = self._format_messages_anthropic(messages, system_msg)

        kwargs = dict(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=formatted,
        )

        # Inject tool definitions so the model can return tool_use blocks
        tools = self._get_tool_definitions_anthropic()
        if tools and self.tool_executor:
            kwargs["tools"] = tools

        tool_calls: list[dict] = []
        current_tool: dict | None = None

        async with client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    if hasattr(event, "content_block") and event.content_block.type == "tool_use":
                        current_tool = {
                            "id": event.content_block.id,
                            "name": event.content_block.name,
                            "_json": "",
                        }
                elif event.type == "content_block_delta":
                    if hasattr(event, "delta"):
                        if event.delta.type == "text_delta":
                            yield {"type": "text", "content": event.delta.text}
                        elif event.delta.type == "input_json_delta" and current_tool is not None:
                            current_tool["_json"] += event.delta.partial_json
                elif event.type == "content_block_stop":
                    if current_tool is not None:
                        try:
                            args = json.loads(current_tool["_json"]) if current_tool["_json"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        tool_calls.append({
                            "id": current_tool["id"],
                            "name": current_tool["name"],
                            "arguments": args,
                        })
                        current_tool = None
                elif event.type == "message_stop":
                    if tool_calls:
                        yield {"type": "tool_calls", "tool_calls": tool_calls}

    async def _stream_openai(
        self, provider_name: str, model_id: str, messages: list[dict],
        system_msg: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        client = self._get_provider_client(provider_name)
        formatted = self._format_messages_openai(messages, system_msg)

        stream = await client.chat.completions.create(
            model=model_id,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    # ── Tool execution ──────────────────────────────────────

    async def _execute_tools(self, tool_calls: list[dict]) -> list[dict]:
        """Execute tool calls and return results."""
        results = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_args = tc.get("arguments", {})
            try:
                # Route skill_ prefixed tools to skill_router
                if tool_name.startswith("skill_") and self.skill_router:
                    real_name = tool_name[6:]  # strip "skill_"
                    skill = self.skill_router.loader.get_skill(real_name)
                    if skill:
                        result = await skill.execute(**tool_args)
                    else:
                        result = {"success": False, "error": f"Skill not found: {real_name}"}
                else:
                    result = await self.tool_executor.execute(tool_name, tool_args)
                results.append({"tool": tool_name, "success": True, "result": result})
            except Exception as e:
                results.append({"tool": tool_name, "success": False, "error": str(e)})
        return results
