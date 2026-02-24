"""
Abstract base class for LLM providers.
Inspired by the trait system of ZeroClaw — every provider implements a
uniform interface so the brain can remain provider-agnostic.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator


class ProviderBase(ABC):
    """Contract that every LLM provider must fulfil."""

    @abstractmethod
    async def generate(
        self,
        model: str,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict:
        """Send a chat completion request and return a result dict.

        The returned dict MUST contain at least::

            {
                "content": str,
                "model": str,
                "usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int},
                "tool_calls": list[dict],   # may be empty
            }
        """
        ...

    @abstractmethod
    async def stream(
        self,
        model: str,
        messages: list[dict],
        system: str,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Yield text chunks as they arrive from the provider."""
        ...
        # Make this a proper async generator
        yield  # pragma: no cover

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider identifier (e.g. 'anthropic', 'openai')."""
        ...

    @property
    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether this provider supports native tool / function calling."""
        ...
