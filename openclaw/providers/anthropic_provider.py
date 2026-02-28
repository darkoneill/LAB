"""
Anthropic (Claude) provider — implements ProviderBase.
Extracted from brain.py _call_anthropic / _stream_anthropic.
"""

import json
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
    ) -> AsyncGenerator[dict, None]:
        """Yield structured events: text deltas and tool_calls."""
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

        tool_calls: list[dict] = []
        current_tool: dict | None = None

        async with client.messages.stream(**kwargs) as stream_ctx:
            async for event in stream_ctx:
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

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
            self._client = anthropic.AsyncAnthropic(api_key=self._api_key)
        return self._client
