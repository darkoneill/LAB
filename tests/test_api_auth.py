"""
Tests for T5: API key authentication.
Covers: APIKeyMiddleware enforcement, setup_wizard key generation,
        SecurityMiddleware.validate_request, /health exemption.
"""

import uuid
from unittest.mock import patch, MagicMock

import pytest
from httpx import AsyncClient, ASGITransport

from openclaw.gateway.server import GatewayServer
from openclaw.gateway.middleware import SecurityMiddleware
from openclaw.setup_wizard import SetupWizard

VALID_KEY = "test-valid-key-1234"


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def gw():
    """GatewayServer with api_key_required=true and one valid key."""
    server = GatewayServer(agent_brain=None, memory_manager=None, skill_router=None)
    server.settings.set("gateway.security.api_key_required", True)
    server.settings.set("gateway.security.api_keys", [VALID_KEY])
    return server


@pytest.fixture
def app(gw):
    return gw.app


@pytest.fixture
def gw_no_auth():
    """GatewayServer with api_key_required=false."""
    server = GatewayServer(agent_brain=None, memory_manager=None, skill_router=None)
    server.settings.set("gateway.security.api_key_required", False)
    return server


@pytest.fixture
def app_no_auth(gw_no_auth):
    return gw_no_auth.app


# ══════════════════════════════════════════════════════════════
#  MIDDLEWARE: routes that MUST require API key
# ══════════════════════════════════════════════════════════════


class TestAPIKeyRequired:

    @pytest.mark.asyncio
    async def test_api_info_rejected_without_key(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/info")
        assert r.status_code == 401
        assert "API key" in r.json()["error"]

    @pytest.mark.asyncio
    async def test_api_sessions_rejected_without_key(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/sessions")
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_api_chat_rejected_without_key(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/api/chat", json={
                "messages": [{"role": "user", "content": "hi"}]
            })
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_api_chat_simple_rejected_without_key(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/api/chat/simple", json={"message": "hi"})
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_api_config_rejected_without_key(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/config")
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_api_models_rejected_without_key(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/models")
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_api_skills_rejected_without_key(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/skills")
        assert r.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_key_rejected(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/info", headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401


# ══════════════════════════════════════════════════════════════
#  MIDDLEWARE: /health MUST be exempt
# ══════════════════════════════════════════════════════════════


class TestHealthExempt:

    @pytest.mark.asyncio
    async def test_health_no_key_required(self, app):
        """GET /health must work without any API key."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_with_key_also_works(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health", headers={"X-API-Key": VALID_KEY})
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════
#  MIDDLEWARE: valid key grants access
# ══════════════════════════════════════════════════════════════


class TestValidKeyAccess:

    @pytest.mark.asyncio
    async def test_valid_key_passes_info(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/info", headers={"X-API-Key": VALID_KEY})
        assert r.status_code == 200
        assert "name" in r.json()

    @pytest.mark.asyncio
    async def test_valid_key_passes_sessions(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/api/sessions", headers={"X-API-Key": VALID_KEY})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_valid_key_passes_chat(self, app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.post("/api/chat", json={
                "messages": [{"role": "user", "content": "hi"}]
            }, headers={"X-API-Key": VALID_KEY})
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_multiple_valid_keys(self, gw):
        """When multiple keys are configured, any one of them should work."""
        second_key = "second-valid-key"
        gw.settings.set("gateway.security.api_keys", [VALID_KEY, second_key])
        async with AsyncClient(transport=ASGITransport(app=gw.app), base_url="http://test") as c:
            r1 = await c.get("/api/info", headers={"X-API-Key": VALID_KEY})
            r2 = await c.get("/api/info", headers={"X-API-Key": second_key})
        assert r1.status_code == 200
        assert r2.status_code == 200


# ══════════════════════════════════════════════════════════════
#  MIDDLEWARE: api_key_required=false disables enforcement
# ══════════════════════════════════════════════════════════════


class TestAuthDisabled:

    @pytest.mark.asyncio
    async def test_no_key_needed_when_disabled(self, app_no_auth):
        async with AsyncClient(transport=ASGITransport(app=app_no_auth), base_url="http://test") as c:
            r = await c.get("/api/info")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_sessions_accessible_when_disabled(self, app_no_auth):
        async with AsyncClient(transport=ASGITransport(app=app_no_auth), base_url="http://test") as c:
            r = await c.get("/api/sessions")
        assert r.status_code == 200


# ══════════════════════════════════════════════════════════════
#  SecurityMiddleware.validate_request (unit-level)
# ══════════════════════════════════════════════════════════════


class TestSecurityMiddlewareValidation:

    def test_valid_key_accepted(self):
        with patch("openclaw.gateway.middleware.get_settings") as mock_gs:
            settings = MagicMock()
            settings.get = lambda k, d=None: {
                "gateway.security.api_key_required": True,
                "gateway.security.api_keys": ["my-key"],
                "gateway.security.max_prompt_length": 32000,
                "gateway.security.content_filtering": False,
            }.get(k, d)
            mock_gs.return_value = settings
            mw = SecurityMiddleware()
            ok, msg = mw.validate_request("hello", api_key="my-key")
        assert ok is True

    def test_invalid_key_rejected(self):
        with patch("openclaw.gateway.middleware.get_settings") as mock_gs:
            settings = MagicMock()
            settings.get = lambda k, d=None: {
                "gateway.security.api_key_required": True,
                "gateway.security.api_keys": ["my-key"],
                "gateway.security.max_prompt_length": 32000,
                "gateway.security.content_filtering": False,
            }.get(k, d)
            mock_gs.return_value = settings
            mw = SecurityMiddleware()
            ok, msg = mw.validate_request("hello", api_key="wrong")
        assert ok is False
        assert "Invalid API key" in msg

    def test_no_key_required(self):
        with patch("openclaw.gateway.middleware.get_settings") as mock_gs:
            settings = MagicMock()
            settings.get = lambda k, d=None: {
                "gateway.security.api_key_required": False,
                "gateway.security.max_prompt_length": 32000,
                "gateway.security.content_filtering": False,
            }.get(k, d)
            mock_gs.return_value = settings
            mw = SecurityMiddleware()
            ok, msg = mw.validate_request("hello", api_key="")
        assert ok is True


# ══════════════════════════════════════════════════════════════
#  Setup Wizard: API key generation
# ══════════════════════════════════════════════════════════════


class TestSetupWizardAPIKey:

    def test_ensure_api_key_generates_uuid(self, tmp_path):
        """_ensure_api_key should generate a uuid4 key when no keys exist."""
        settings = MagicMock()
        settings.get = MagicMock(return_value=[])
        settings.set = MagicMock()

        wizard = SetupWizard()
        wizard.settings = settings
        wizard._ensure_api_key()

        # Should have called set with a list containing one uuid4 string
        settings.set.assert_called_once()
        call_args = settings.set.call_args
        assert call_args[0][0] == "gateway.security.api_keys"
        key_list = call_args[0][1]
        assert isinstance(key_list, list)
        assert len(key_list) == 1
        # Verify it's a valid uuid4
        parsed = uuid.UUID(key_list[0], version=4)
        assert str(parsed) == key_list[0]
        # Should persist
        assert call_args[1].get("persist") is True or call_args.kwargs.get("persist") is True
        # Should store as attribute for display
        assert hasattr(wizard, "_generated_api_key")
        assert wizard._generated_api_key == key_list[0]

    def test_ensure_api_key_skips_when_keys_exist(self):
        """_ensure_api_key should not overwrite existing keys."""
        settings = MagicMock()
        settings.get = MagicMock(return_value=["existing-key-123"])
        settings.set = MagicMock()

        wizard = SetupWizard()
        wizard.settings = settings
        wizard._ensure_api_key()

        settings.set.assert_not_called()
        assert not hasattr(wizard, "_generated_api_key")

    def test_generated_key_is_unique(self):
        """Two calls should produce different keys."""
        settings1 = MagicMock()
        settings1.get = MagicMock(return_value=[])
        settings1.set = MagicMock()

        settings2 = MagicMock()
        settings2.get = MagicMock(return_value=[])
        settings2.set = MagicMock()

        w1 = SetupWizard()
        w1.settings = settings1
        w1._ensure_api_key()

        w2 = SetupWizard()
        w2.settings = settings2
        w2._ensure_api_key()

        assert w1._generated_api_key != w2._generated_api_key


# ══════════════════════════════════════════════════════════════
#  default.yaml: api_key_required should be true
# ══════════════════════════════════════════════════════════════


class TestDefaultConfig:

    def test_api_key_required_enabled_by_default(self):
        """Verify default.yaml has api_key_required: true."""
        import yaml
        from pathlib import Path
        default_yaml = Path(__file__).parent.parent / "openclaw" / "config" / "default.yaml"
        with open(default_yaml) as f:
            cfg = yaml.safe_load(f)
        assert cfg["gateway"]["security"]["api_key_required"] is True
