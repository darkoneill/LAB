"""
Tests for new providers: Gemini, OpenRouter, and Ollama tool support.
All network I/O is mocked.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw.providers.factory import create_provider, _ollama_supports_tools


# ── Gemini Provider ──────────────────────────────────────────


class TestGeminiGenerate:
    @pytest.mark.asyncio
    async def test_formats_request_and_parses_response(self):
        from openclaw.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider(api_key="fake-key")

        gemini_response = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello from Gemini!"}],
                    "role": "model",
                },
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = gemini_response
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("openclaw.providers.gemini_provider.httpx.AsyncClient", return_value=mock_client):
            result = await provider.generate(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": "Hi"}],
                system="You are helpful.",
            )

        assert result["content"] == "Hello from Gemini!"
        assert result["model"] == "gemini-2.0-flash"
        assert result["usage"]["total_tokens"] == 15
        assert result["tool_calls"] == []

        # Verify request format
        call_kwargs = mock_client.post.call_args
        body = call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        assert "contents" in body
        assert body["contents"][0]["parts"][0]["text"] == "Hi"
        assert body["systemInstruction"]["parts"][0]["text"] == "You are helpful."

    @pytest.mark.asyncio
    async def test_parses_function_calls(self):
        from openclaw.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider(api_key="fake-key")

        gemini_response = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Let me search."},
                        {"functionCall": {"name": "web_search", "args": {"query": "test"}}},
                    ],
                    "role": "model",
                },
            }],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10, "totalTokenCount": 15},
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = gemini_response
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("openclaw.providers.gemini_provider.httpx.AsyncClient", return_value=mock_client):
            result = await provider.generate(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": "search test"}],
                system="",
            )

        assert result["content"] == "Let me search."
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "web_search"
        assert result["tool_calls"][0]["arguments"] == {"query": "test"}

    @pytest.mark.asyncio
    async def test_formats_tools(self):
        from openclaw.providers.gemini_provider import GeminiProvider

        tools = [
            {"function": {"name": "shell", "description": "Run shell", "parameters": {"type": "object"}}},
        ]
        result = GeminiProvider._format_tools(tools)
        assert result is not None
        assert len(result) == 1
        assert result[0]["functionDeclarations"][0]["name"] == "shell"

    def test_format_tools_none(self):
        from openclaw.providers.gemini_provider import GeminiProvider

        assert GeminiProvider._format_tools(None) is None
        assert GeminiProvider._format_tools([]) is None


class TestGeminiStream:
    @pytest.mark.asyncio
    async def test_yields_text_events(self):
        from openclaw.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider(api_key="fake-key")

        sse_lines = [
            'data: {"candidates":[{"content":{"parts":[{"text":"Hello "}]}}]}',
            'data: {"candidates":[{"content":{"parts":[{"text":"world!"}]}}]}',
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_stream_resp = MagicMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.aiter_lines = mock_aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("openclaw.providers.gemini_provider.httpx.AsyncClient", return_value=mock_client):
            events = []
            async for event in provider.stream(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": "Hi"}],
                system="",
            ):
                events.append(event)

        assert len(events) == 2
        assert events[0] == {"type": "text", "content": "Hello "}
        assert events[1] == {"type": "text", "content": "world!"}

    @pytest.mark.asyncio
    async def test_stream_with_function_calls(self):
        from openclaw.providers.gemini_provider import GeminiProvider

        provider = GeminiProvider(api_key="fake-key")

        sse_lines = [
            'data: {"candidates":[{"content":{"parts":[{"text":"Searching..."}]}}]}',
            'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{"q":"test"}}}]}}]}',
        ]

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_stream_resp = MagicMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.aiter_lines = mock_aiter_lines
        mock_stream_resp.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        mock_stream_resp.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("openclaw.providers.gemini_provider.httpx.AsyncClient", return_value=mock_client):
            events = []
            async for event in provider.stream(
                model="gemini-2.0-flash",
                messages=[{"role": "user", "content": "search"}],
                system="",
            ):
                events.append(event)

        assert events[0] == {"type": "text", "content": "Searching..."}
        assert events[1]["type"] == "tool_calls"
        assert events[1]["tool_calls"][0]["name"] == "search"


class TestGeminiProperties:
    def test_name(self):
        from openclaw.providers.gemini_provider import GeminiProvider

        assert GeminiProvider(api_key="x").name == "gemini"

    def test_supports_tools(self):
        from openclaw.providers.gemini_provider import GeminiProvider

        assert GeminiProvider(api_key="x").supports_tools is True


# ── OpenRouter Provider ──────────────────────────────────────


class TestOpenRouterProvider:
    def test_uses_openai_compat(self):
        from openclaw.providers.openrouter_provider import OpenRouterProvider

        p = OpenRouterProvider(api_key="or-key")
        assert p.name == "openrouter"
        assert p._base_url == "https://openrouter.ai/api/v1"
        assert p.supports_tools is True

    def test_adds_custom_headers(self):
        from openclaw.providers.openrouter_provider import OpenRouterProvider

        p = OpenRouterProvider(
            api_key="or-key",
            http_referer="https://myapp.com",
            x_title="MyApp",
        )
        headers = p._extra_headers()
        assert headers["HTTP-Referer"] == "https://myapp.com"
        assert headers["X-Title"] == "MyApp"

    def test_empty_referer_omitted(self):
        from openclaw.providers.openrouter_provider import OpenRouterProvider

        p = OpenRouterProvider(api_key="or-key", http_referer="")
        headers = p._extra_headers()
        assert "HTTP-Referer" not in headers

    def test_get_client_passes_headers(self):
        from openclaw.providers.openrouter_provider import OpenRouterProvider

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI.return_value = MagicMock()

        p = OpenRouterProvider(
            api_key="or-key",
            http_referer="https://ref.com",
            x_title="Title",
        )

        with patch.dict("sys.modules", {"openai": mock_openai}):
            p._get_client()

        call_kwargs = mock_openai.AsyncOpenAI.call_args[1]
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert call_kwargs["default_headers"]["HTTP-Referer"] == "https://ref.com"
        assert call_kwargs["default_headers"]["X-Title"] == "Title"


# ── Ollama Tool Support Detection ────────────────────────────


class TestOllamaToolSupport:
    def test_llama31_has_tools(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "providers.ollama.default_model": "llama3.1:70b-instruct",
            "providers.ollama.tool_capable_models": [],
        }.get(k, d)
        assert _ollama_supports_tools(settings) is True

    def test_llama32_has_tools(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "providers.ollama.default_model": "llama3.2",
            "providers.ollama.tool_capable_models": [],
        }.get(k, d)
        assert _ollama_supports_tools(settings) is True

    def test_qwen25_has_tools(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "providers.ollama.default_model": "qwen2.5:7b",
            "providers.ollama.tool_capable_models": [],
        }.get(k, d)
        assert _ollama_supports_tools(settings) is True

    def test_mistral_nemo_has_tools(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "providers.ollama.default_model": "mistral-nemo",
            "providers.ollama.tool_capable_models": [],
        }.get(k, d)
        assert _ollama_supports_tools(settings) is True

    def test_old_model_no_tools(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "providers.ollama.default_model": "llama2",
            "providers.ollama.tool_capable_models": [],
        }.get(k, d)
        assert _ollama_supports_tools(settings) is False

    def test_custom_model_in_list(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "providers.ollama.default_model": "my-custom-model",
            "providers.ollama.tool_capable_models": ["my-custom-model"],
        }.get(k, d)
        assert _ollama_supports_tools(settings) is True

    def test_factory_creates_ollama_with_tools(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "providers.ollama.base_url": "http://localhost:11434",
            "providers.ollama.default_model": "llama3.1",
            "providers.ollama.tool_capable_models": [],
        }.get(k, d)
        provider = create_provider("ollama", settings)
        assert provider.supports_tools is True

    def test_factory_creates_ollama_without_tools(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "providers.ollama.base_url": "http://localhost:11434",
            "providers.ollama.default_model": "llama2",
            "providers.ollama.tool_capable_models": [],
        }.get(k, d)
        provider = create_provider("ollama", settings)
        assert provider.supports_tools is False


# ── Factory new providers ────────────────────────────────────


class TestFactoryNewProviders:
    def test_create_gemini(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {"providers.gemini.api_key": "gk"}.get(k, d)
        p = create_provider("gemini", settings)
        assert p.name == "gemini"
        assert p._api_key == "gk"

    def test_create_openrouter(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "providers.openrouter.api_key": "ork",
            "providers.openrouter.http_referer": "https://app.com",
            "providers.openrouter.x_title": "MyApp",
        }.get(k, d)
        p = create_provider("openrouter", settings)
        assert p.name == "openrouter"
        assert p._http_referer == "https://app.com"

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("nonexistent", MagicMock())
