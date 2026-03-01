"""
Google Gemini provider — implements ProviderBase using httpx (no SDK).

Endpoint: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
Auth: API key as query param ``?key=``
"""

import json
import logging
import os
from typing import AsyncGenerator

import httpx

from openclaw.providers.base import ProviderBase

logger = logging.getLogger("openclaw.providers.gemini")

_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiProvider(ProviderBase):
    """Google Gemini provider via REST (no google-genai SDK)."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def supports_tools(self) -> bool:
        return True

    # ── Message formatting ───────────────────────────────

    @staticmethod
    def _format_contents(messages: list[dict], system: str) -> tuple[list[dict], dict | None]:
        """Convert chat messages to Gemini ``contents`` format.

        Returns (contents, system_instruction | None).
        """
        sys_instruction = {"parts": [{"text": system}]} if system else None

        contents: list[dict] = []
        for msg in messages:
            role = "model" if msg.get("role") == "assistant" else "user"
            text = msg.get("content", "")
            if isinstance(text, str):
                contents.append({"role": role, "parts": [{"text": text}]})
            elif isinstance(text, list):
                # Multi-block content (tool results, etc.)
                parts = []
                for block in text:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append({"text": block["text"]})
                    elif isinstance(block, dict) and block.get("type") == "tool_result":
                        parts.append({
                            "functionResponse": {
                                "name": block.get("tool_use_id", ""),
                                "response": {"content": json.dumps(block.get("content", ""))},
                            }
                        })
                    else:
                        parts.append({"text": str(block)})
                if parts:
                    contents.append({"role": role, "parts": parts})
        return contents, sys_instruction

    @staticmethod
    def _format_tools(tools: list[dict] | None) -> list[dict] | None:
        """Convert OpenAI-style tool defs to Gemini ``functionDeclarations``."""
        if not tools:
            return None
        declarations = []
        for tool in tools:
            fn = tool.get("function", tool)
            declarations.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        return [{"functionDeclarations": declarations}]

    # ── API calls ────────────────────────────────────────

    async def generate(
        self,
        model: str,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict:
        contents, sys_instruction = self._format_contents(messages, system)

        body: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if sys_instruction:
            body["systemInstruction"] = sys_instruction
        gemini_tools = self._format_tools(tools)
        if gemini_tools:
            body["tools"] = gemini_tools

        url = f"{_API_BASE}/models/{model}:generateContent"

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, params={"key": self._api_key}, json=body)
            resp.raise_for_status()
            data = resp.json()

        return self._parse_response(data, model)

    async def stream(
        self,
        model: str,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[dict, None]:
        contents, sys_instruction = self._format_contents(messages, system)

        body: dict = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if sys_instruction:
            body["systemInstruction"] = sys_instruction
        gemini_tools = self._format_tools(tools)
        if gemini_tools:
            body["tools"] = gemini_tools

        url = f"{_API_BASE}/models/{model}:streamGenerateContent"

        tool_calls: list[dict] = []

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", url, params={"key": self._api_key, "alt": "sse"}, json=body
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if raw.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    for candidate in chunk.get("candidates", []):
                        for part in candidate.get("content", {}).get("parts", []):
                            if "text" in part:
                                yield {"type": "text", "content": part["text"]}
                            elif "functionCall" in part:
                                fc = part["functionCall"]
                                tool_calls.append({
                                    "id": f"gemini_{fc['name']}",
                                    "name": fc["name"],
                                    "arguments": fc.get("args", {}),
                                })

        if tool_calls:
            yield {"type": "tool_calls", "tool_calls": tool_calls}

    # ── Response parsing ─────────────────────────────────

    @staticmethod
    def _parse_response(data: dict, model: str) -> dict:
        content = ""
        tool_calls: list[dict] = []

        for candidate in data.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                if "text" in part:
                    content += part["text"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    tool_calls.append({
                        "id": f"gemini_{fc['name']}",
                        "name": fc["name"],
                        "arguments": fc.get("args", {}),
                    })

        usage_meta = data.get("usageMetadata", {})
        return {
            "content": content,
            "model": model,
            "usage": {
                "input_tokens": usage_meta.get("promptTokenCount", 0),
                "output_tokens": usage_meta.get("candidatesTokenCount", 0),
                "total_tokens": usage_meta.get("totalTokenCount", 0),
            },
            "tool_calls": tool_calls,
        }
