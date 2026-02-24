"""
Anthropic (Claude) provider — implements ProviderBase.
Extracted from brain.py _call_anthropic / _stream_anthropic.
"""

import logging
import os
from typing import AsyncGenerator

from openclaw.providers.base import ProviderBase

logger = logging.getLogger("openclaw.providers.anthropic")


class AnthropicProvider(ProviderBase):
    """Native Anthropic Messages API provider."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._client = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def supports_tools(self) -> bool:
        return True

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

        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        if tools:
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
            "model": model,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
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
    ) -> AsyncGenerator[str, None]:
        client = self._get_client()

        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        ) as stream_ctx:
            async for text in stream_ctx.text_stream:
                yield text

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client
