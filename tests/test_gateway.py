"""
Gateway API tests - minimal suite covering core routes and security.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from openclaw.gateway.server import GatewayServer

TEST_API_KEY = "test-key-for-gateway-tests"


@pytest.fixture
def app():
    gw = GatewayServer(agent_brain=None, memory_manager=None, skill_router=None)
    # Inject a test API key so /api/* routes pass auth
    gw.settings.set("gateway.security.api_keys", [TEST_API_KEY])
    return gw.app


@pytest.fixture
def auth_headers():
    return {"X-API-Key": TEST_API_KEY}


@pytest.mark.asyncio
async def test_health(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_info(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/info", headers=auth_headers)
    assert r.status_code == 200
    assert "name" in r.json()


@pytest.mark.asyncio
async def test_list_sessions_empty(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/sessions", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["sessions"] == []


@pytest.mark.asyncio
async def test_create_session(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/sessions", json={}, headers=auth_headers)
    assert r.status_code == 200
    assert "id" in r.json()


@pytest.mark.asyncio
async def test_session_history_404(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/sessions/nonexistent/history", headers=auth_headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_session_history_empty_session(app, auth_headers):
    """Session exists but no messages -> should return 200 with empty list, not 404."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        cr = await client.post("/api/sessions", json={}, headers=auth_headers)
        session_id = cr.json()["id"]
        r = await client.get(f"/api/sessions/{session_id}/history", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["messages"] == []


@pytest.mark.asyncio
async def test_message_too_large(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "x" * 50_000}]
        }, headers=auth_headers)
    assert r.status_code == 422  # pydantic validation error


@pytest.mark.asyncio
async def test_too_many_messages(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "hi"}] * 201
        }, headers=auth_headers)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_chat_simple_too_large(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/chat/simple", json={
            "message": "x" * 50_000
        }, headers=auth_headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_chat_simple_empty(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/chat/simple", json={"message": ""}, headers=auth_headers)
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_chat_no_agent(app, auth_headers):
    """When agent_brain is None, chat should return a helpful message, not crash."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "hello"}]
        }, headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert "not initialized" in body.get("content", "").lower() or "configure" in body.get("content", "").lower()


@pytest.mark.asyncio
async def test_config_get(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/config", headers=auth_headers)
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


@pytest.mark.asyncio
async def test_config_put_blocked_key(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.put("/api/config", json={
            "gateway.admin_key": "hacked"
        }, headers=auth_headers)
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_delete_session_404(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.delete("/api/sessions/nonexistent", headers=auth_headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_skills_empty(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/skills", headers=auth_headers)
    assert r.status_code == 200
    assert "skills" in r.json()


@pytest.mark.asyncio
async def test_models_list(app, auth_headers):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/models", headers=auth_headers)
    assert r.status_code == 200
    assert "models" in r.json()
