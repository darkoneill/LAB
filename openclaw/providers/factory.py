"""
Provider factory — instantiates the right ProviderBase from config.
"""

import logging

from openclaw.providers.base import ProviderBase

logger = logging.getLogger("openclaw.providers.factory")

# Ollama models known to support tool/function calling.
_OLLAMA_TOOL_CAPABLE = {"llama3.1", "llama3.2", "llama3.3", "qwen2.5", "mistral-nemo"}


def _ollama_supports_tools(settings) -> bool:
    """Return True if the configured Ollama model is tool-capable."""
    model = settings.get("providers.ollama.default_model", "")
    # Also accept a user-defined list
    custom = settings.get("providers.ollama.tool_capable_models", [])
    known = _OLLAMA_TOOL_CAPABLE | set(custom)
    # Match prefix: "llama3.1:70b-instruct" starts with "llama3.1"
    return any(model.startswith(prefix) for prefix in known)


def create_provider(name: str, settings) -> ProviderBase:
    """Create a concrete provider instance from its config name.

    Args:
        name: One of 'anthropic', 'openai', 'ollama', 'custom',
              'gemini', 'openrouter'.
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
        tool_support = _ollama_supports_tools(settings)
        return OpenAIProvider(
            api_key="ollama",
            base_url=f"{base_url}/v1",
            provider_name="ollama",
            tool_support=tool_support,
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

    if name == "gemini":
        from openclaw.providers.gemini_provider import GeminiProvider

        api_key = settings.get("providers.gemini.api_key", "")
        return GeminiProvider(api_key=api_key)

    if name == "openrouter":
        from openclaw.providers.openrouter_provider import OpenRouterProvider

        api_key = settings.get("providers.openrouter.api_key", "")
        http_referer = settings.get("providers.openrouter.http_referer", "")
        x_title = settings.get("providers.openrouter.x_title", "OpenClaw")
        return OpenRouterProvider(
            api_key=api_key,
            http_referer=http_referer,
            x_title=x_title,
        )

    raise ValueError(f"Unknown provider: {name}")
