"""
Provider factory — instantiates the right ProviderBase from config.
"""

import logging

from openclaw.providers.base import ProviderBase

logger = logging.getLogger("openclaw.providers.factory")


def create_provider(name: str, settings) -> ProviderBase:
    """Create a concrete provider instance from its config name.

    Args:
        name: One of 'anthropic', 'openai', 'ollama', 'custom'.
        settings: A Settings-like object with ``.get(dotpath, default)``.

    Returns:
        A fully initialised ProviderBase instance.

    Raises:
        ValueError: If the provider name is unknown.
    """
    if name == "anthropic":
        from openclaw.providers.anthropic_provider import AnthropicProvider

        api_key = settings.get("providers.anthropic.api_key", "")
        return AnthropicProvider(api_key=api_key)

    if name == "openai":
        from openclaw.providers.openai_provider import OpenAIProvider

        api_key = settings.get("providers.openai.api_key", "")
        base_url = settings.get("providers.openai.base_url", None) or None
        return OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            provider_name="openai",
            tool_support=True,
        )

    if name == "ollama":
        from openclaw.providers.openai_provider import OpenAIProvider

        base_url = settings.get("providers.ollama.base_url", "http://localhost:11434")
        return OpenAIProvider(
            api_key="ollama",
            base_url=f"{base_url}/v1",
            provider_name="ollama",
            tool_support=False,
        )

    if name == "custom":
        from openclaw.providers.openai_provider import OpenAIProvider

        api_key = settings.get("providers.custom.api_key", "")
        base_url = settings.get("providers.custom.base_url", "")
        return OpenAIProvider(
            api_key=api_key,
            base_url=base_url or None,
            provider_name="custom",
            tool_support=True,
        )

    raise ValueError(f"Unknown provider: {name}")
