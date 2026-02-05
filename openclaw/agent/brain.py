"""
Agent Brain - The reasoning engine.
Multi-model, prompt-driven LLM layer inspired by AgentZero's architecture.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional, AsyncGenerator

from openclaw.config.settings import get_settings
from openclaw.gateway.router import RequestRouter

logger = logging.getLogger("openclaw.agent.brain")


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

        return "\n".join(parts)

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

        response = await client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=formatted,
        )

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return {
            "content": content,
            "model": model_id,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            "tool_calls": [],
        }

    async def _call_openai_compat(
        self, provider_name: str, model_id: str, messages: list[dict],
        system_msg: str, temperature: float, max_tokens: int
    ) -> dict:
        client = self._get_provider_client(provider_name)
        formatted = self._format_messages_openai(messages, system_msg)

        response = await client.chat.completions.create(
            model=model_id,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        usage = response.usage

        return {
            "content": choice.message.content or "",
            "model": model_id,
            "usage": {
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "tool_calls": [],
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
                result = await self.tool_executor.execute(tool_name, tool_args)
                results.append({"tool": tool_name, "success": True, "result": result})
            except Exception as e:
                results.append({"tool": tool_name, "success": False, "error": str(e)})
        return results
