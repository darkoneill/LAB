"""
OpenRouter provider — OpenAI-compatible with custom headers.

Composes with :class:`OpenAIProvider` (same wire format), adding the
``HTTP-Referer`` and ``X-Title`` headers that OpenRouter requires.
"""

import logging
import os

from openclaw.providers.openai_provider import OpenAIProvider

logger = logging.getLogger("openclaw.providers.openrouter")

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter.ai provider (OpenAI-compatible + custom headers)."""

    def __init__(
        self,
        api_key: str = "",
        http_referer: str = "",
        x_title: str = "OpenClaw",
    ):
        super().__init__(
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
            base_url=_OPENROUTER_BASE,
            provider_name="openrouter",
            tool_support=True,
        )
        self._http_referer = http_referer
        self._x_title = x_title

    @property
    def name(self) -> str:
        return "openrouter"

    def _get_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                default_headers=self._extra_headers(),
            )
        return self._client

    def _extra_headers(self) -> dict:
        headers = {}
        if self._http_referer:
            headers["HTTP-Referer"] = self._http_referer
        if self._x_title:
            headers["X-Title"] = self._x_title
        return headers
