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
    """

    def __init__(self, tool_executor=None, skill_router=None):
        self.settings = get_settings()
        self.router = RequestRouter()
        self.tool_executor = tool_executor
        self.skill_router = skill_router
        self._system_prompt = ""
        self._personality = ""
        self._tools_prompt = ""
        self._load_prompts()
        self._tool_definitions = None  # cached tool defs

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

    def _get_provider_client(self, provider_name: str):
        """Get an LLM client for the specified provider."""
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

    def _format_messages_anthropic(self, messages: list[dict], system_msg: str) -> tuple[str, list[dict]]:
        """Format messages for Anthropic API."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                continue
            if role not in ("user", "assistant"):
                role = "user"
            # Avoid consecutive same-role messages
            if formatted and formatted[-1]["role"] == role:
                formatted[-1]["content"] += f"\n{content}"
            else:
                formatted.append({"role": role, "content": content})

        # Ensure first message is from user
        if not formatted or formatted[0]["role"] != "user":
            formatted.insert(0, {"role": "user", "content": "(conversation start)"})

        return system_msg, formatted

    def _format_messages_openai(self, messages: list[dict], system_msg: str) -> list[dict]:
        """Format messages for OpenAI-compatible API."""
        formatted = [{"role": "system", "content": system_msg}]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role not in ("system", "user", "assistant"):
                role = "user"
            formatted.append({"role": role, "content": content})
        return formatted

    async def generate(
        self,
        messages: list[dict],
        memory_context: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        iteration: int = 0,
    ) -> dict:
        """Generate a complete response with optional tool execution."""
        max_iter = self.settings.get("agent.max_iterations", 25)
        if iteration >= max_iter:
            return {"content": "[Max iterations reached]", "model": "", "usage": {}}

        temp = temperature or self.settings.get("agent.temperature", 0.7)
        max_tok = max_tokens or self.settings.get("agent.max_tokens", 4096)
        system_msg = self._build_system_message(memory_context)

        provider_name, model_id = self.router.resolve_model(model)

        try:
            start_time = time.time()
            result = await self._call_provider(provider_name, model_id, messages, system_msg, temp, max_tok)
            latency = (time.time() - start_time) * 1000
            self.router.record_success(provider_name, latency, result.get("usage", {}).get("total_tokens", 0))

            # Extract and log thinking
            if self.settings.get("agent.chain_of_thought", True):
                thinking, clean_content = self._extract_thinking(result["content"])
                if thinking:
                    logger.debug(f"Agent thinking: {thinking[:200]}...")
                    result["thinking"] = thinking
                    result["content"] = clean_content

            # Handle tool calls
            if result.get("tool_calls") and self.tool_executor:
                tool_results = await self._execute_tools(result["tool_calls"])
                messages = messages + [
                    {"role": "assistant", "content": result["content"]},
                    {"role": "user", "content": f"Tool results:\n{json.dumps(tool_results, indent=2)}"},
                ]
                return await self.generate(messages, memory_context, model, temperature, max_tokens, iteration + 1)

            return result

        except Exception as e:
            logger.error(f"Provider {provider_name} failed: {e}")
            self.router.record_failure(provider_name)

            # Try failover
            failover = self.router.get_failover(provider_name, model_id)
            if failover:
                fo_provider, fo_model = failover
                try:
                    start_time = time.time()
                    result = await self._call_provider(fo_provider, fo_model, messages, system_msg, temp, max_tok)
                    latency = (time.time() - start_time) * 1000
                    self.router.record_success(fo_provider, latency)
                    return result
                except Exception as e2:
                    logger.error(f"Failover to {fo_provider} also failed: {e2}")

            return {"content": f"Error: {str(e)}", "model": model_id, "usage": {}, "tool_calls": []}

    async def generate_stream(
        self,
        messages: list[dict],
        memory_context: str = "",
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens."""
        temp = temperature or self.settings.get("agent.temperature", 0.7)
        max_tok = max_tokens or self.settings.get("agent.max_tokens", 4096)
        system_msg = self._build_system_message(memory_context)
        provider_name, model_id = self.router.resolve_model(model)

        try:
            if provider_name == "anthropic":
                async for chunk in self._stream_anthropic(model_id, messages, system_msg, temp, max_tok):
                    yield chunk
            else:
                async for chunk in self._stream_openai(provider_name, model_id, messages, system_msg, temp, max_tok):
                    yield chunk
        except Exception as e:
            logger.error(f"Streaming error with {provider_name}: {e}")
            yield f"\n[Error: {str(e)}]"

    async def _call_provider(
        self, provider_name: str, model_id: str, messages: list[dict],
        system_msg: str, temperature: float, max_tokens: int
    ) -> dict:
        """Call a specific provider."""
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
    ) -> AsyncGenerator[str, None]:
        client = self._get_anthropic_client()
        system, formatted = self._format_messages_anthropic(messages, system_msg)

        async with client.messages.stream(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=formatted,
        ) as stream:
            async for text in stream.text_stream:
                yield text

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
