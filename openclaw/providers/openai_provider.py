"""
OpenAI-compatible provider — implements ProviderBase.
Works with OpenAI, Ollama (/v1), and any OpenAI-compatible endpoint.
Extracted from brain.py _call_openai_compat / _stream_openai.
"""

import json
import logging
import os
from typing import AsyncGenerator

from openclaw.providers.base import ProviderBase

logger = logging.getLogger("openclaw.providers.openai")


class OpenAIProvider(ProviderBase):
    """Provider for OpenAI and any OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str | None = None,
        provider_name: str = "openai",
        tool_support: bool = True,
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url or None
        self._provider_name = provider_name
        self._tool_support = tool_support
        self._client = None

    @property
    def name(self) -> str:
        return self._provider_name

    @property
    def supports_tools(self) -> bool:
        return self._tool_support

    async def generate(
        self,
        model: str,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict:
        client = self._get_client()

        # Prepend system message (OpenAI convention)
        formatted = [{"role": "system", "content": system}] + messages

        kwargs = dict(
            model=model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if tools and self._tool_support:
            kwargs["tools"] = tools

        response = await client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        usage = response.usage

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
            "model": model,
            "usage": {
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "tool_calls": tool_calls,
        }

    async def stream(
        self,
        model: str,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[dict, None]:
        """Yield structured text events (OpenAI streaming has no tool_use events)."""
        client = self._get_client()

        formatted = [{"role": "system", "content": system}] + messages

        response = await client.chat.completions.create(
            model=model,
            messages=formatted,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield {"type": "text", "content": chunk.choices[0].delta.content}

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client
