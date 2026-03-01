"""
Tests for openclaw/mcp/ — client.py, server.py, registry.py.
All I/O is mocked: no subprocesses, no network.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from openclaw.mcp.client import MCPClient
from openclaw.mcp.server import MCPServer
from openclaw.mcp.registry import MCPRegistry


# ── Helpers ──────────────────────────────────────────────────


def _stdio_config(name="test-server"):
    return {
        "name": name,
        "transport": "stdio",
        "command": "/usr/bin/fake-mcp",
        "args": ["--json"],
        "env": {},
    }


def _sse_config(name="sse-server"):
    return {
        "name": name,
        "transport": "sse",
        "url": "http://localhost:9999",
    }


def _jsonrpc_result(request_id, result):
    return json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}) + "\n"


def _init_response(request_id=1):
    return _jsonrpc_result(request_id, {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "serverInfo": {"name": "mock", "version": "0.1"},
    })


def _tools_list_response(request_id, tools):
    return _jsonrpc_result(request_id, {"tools": tools})


def _tool_call_response(request_id, text, is_error=False):
    return _jsonrpc_result(request_id, {
        "content": [{"type": "text", "text": text}],
        "isError": is_error,
    })


SAMPLE_TOOLS = [
    {"name": "read", "description": "Read a file", "inputSchema": {"type": "object"}},
    {"name": "write", "description": "Write a file", "inputSchema": {"type": "object"}},
]


# ── MCPClient Tests ─────────────────────────────────────────


class TestMCPClientConnectStdio:
    """Verify stdio subprocess wiring without launching a real process."""

    @pytest.mark.asyncio
    async def test_connect_stdio_starts_subprocess(self):
        """subprocess.Popen is called with the right command, stdin/stdout PIPE."""
        client = MCPClient(_stdio_config())

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.stdout.readline = MagicMock(return_value=b"")

        with patch("openclaw.mcp.client.subprocess.Popen", return_value=mock_proc) as popen:
            # _initialize will be called, which sends requests — mock _request
            with patch.object(client, "_initialize", new_callable=AsyncMock):
                await client._connect_stdio()

        popen.assert_called_once()
        call_kwargs = popen.call_args[1]
        assert call_kwargs["stdin"] is not None  # subprocess.PIPE
        assert call_kwargs["stdout"] is not None

    @pytest.mark.asyncio
    async def test_connect_stdio_sets_process(self):
        client = MCPClient(_stdio_config())
        mock_proc = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = MagicMock(return_value=b"")

        with patch("openclaw.mcp.client.subprocess.Popen", return_value=mock_proc):
            with patch.object(client, "_initialize", new_callable=AsyncMock):
                await client._connect_stdio()

        assert client._process is mock_proc


class TestMCPClientConnectSSE:
    @pytest.mark.asyncio
    async def test_connect_sse_creates_http_client(self):
        client = MCPClient(_sse_config())

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = MagicMock()

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch.object(client, "_initialize", new_callable=AsyncMock):
                await client._connect_sse()

        assert hasattr(client, "_http_client")
        assert client._sse_url == "http://localhost:9999"

    @pytest.mark.asyncio
    async def test_connect_sse_no_url_raises(self):
        client = MCPClient({"name": "bad", "transport": "sse"})

        with pytest.raises(ValueError, match="No URL"):
            await client._connect_sse()


class TestMCPClientSendAndReceive:
    @pytest.mark.asyncio
    async def test_send_writes_json_to_stdin(self):
        client = MCPClient(_stdio_config())
        mock_stdin = MagicMock()
        client._process = MagicMock()
        client._process.stdin = mock_stdin

        msg = {"jsonrpc": "2.0", "id": 1, "method": "test", "params": {}}
        await client._send(msg)

        written = mock_stdin.write.call_args[0][0]
        assert json.loads(written.decode()) == msg

    @pytest.mark.asyncio
    async def test_request_sets_up_future_and_increments_id(self):
        client = MCPClient(_stdio_config())
        client._connected = True
        client._process = MagicMock()
        client._process.stdin = MagicMock()

        # Pre-resolve the future from a "background reader"
        original_send = client._send

        async def fake_send(message):
            await original_send(message)
            rid = message["id"]
            if rid in client._pending_requests:
                client._pending_requests[rid].set_result(
                    {"jsonrpc": "2.0", "id": rid, "result": {"ok": True}}
                )

        with patch.object(client, "_send", side_effect=fake_send):
            resp = await client._request("tools/list", {})

        assert resp["result"]["ok"] is True
        assert client._request_id >= 1

    @pytest.mark.asyncio
    async def test_request_timeout_raises(self):
        """_request raises TimeoutError when the server doesn't respond."""
        client = MCPClient(_stdio_config())
        client._connected = True
        client._process = MagicMock()
        client._process.stdin = MagicMock()

        # Patch the internal timeout to something short
        with patch("openclaw.mcp.client.asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with patch.object(client, "_send", new_callable=AsyncMock):
                with pytest.raises(TimeoutError, match="timed out"):
                    await client._request("tools/list", {})


class TestMCPClientListTools:
    @pytest.mark.asyncio
    async def test_discover_tools_populates_list(self):
        client = MCPClient(_stdio_config())

        async def fake_request(method, params):
            if method == "tools/list":
                return {"result": {"tools": SAMPLE_TOOLS}}
            return {"result": {}}

        with patch.object(client, "_request", side_effect=fake_request):
            await client._discover_tools()

        assert len(client.get_tools()) == 2
        assert client.tool_count == 2
        assert client.get_tools()[0]["name"] == "read"

    @pytest.mark.asyncio
    async def test_get_tools_for_llm_format(self):
        client = MCPClient(_stdio_config())
        client._tools = SAMPLE_TOOLS

        formatted = client.get_tools_for_llm()
        assert len(formatted) == 2
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "test-server_read"


class TestMCPClientCallTool:
    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        client = MCPClient(_stdio_config())
        client._connected = True

        async def fake_request(method, params):
            assert method == "tools/call"
            assert params["name"] == "read"
            return {"result": {
                "content": [{"type": "text", "text": "file contents"}],
                "isError": False,
            }}

        with patch.object(client, "_request", side_effect=fake_request):
            result = await client.call_tool("read", {"path": "/tmp/x"})

        assert result["success"] is True
        assert result["content"] == "file contents"

    @pytest.mark.asyncio
    async def test_call_tool_error_response(self):
        client = MCPClient(_stdio_config())
        client._connected = True

        async def fake_request(method, params):
            return {"error": {"code": -1, "message": "boom"}}

        with patch.object(client, "_request", side_effect=fake_request):
            result = await client.call_tool("bad", {})

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_call_tool_not_connected_raises(self):
        client = MCPClient(_stdio_config())
        client._connected = False

        with pytest.raises(RuntimeError, match="Not connected"):
            await client.call_tool("read", {})


class TestMCPClientInvalidJSON:
    @pytest.mark.asyncio
    async def test_invalid_json_line_skipped(self):
        """_read_responses should skip lines that aren't valid JSON."""
        client = MCPClient(_stdio_config())
        mock_stdout = MagicMock()
        lines = iter([b"not json\n", b""])
        mock_stdout.readline = MagicMock(side_effect=lines)

        client._process = MagicMock()
        client._process.stdout = mock_stdout

        # Should complete without raising
        await client._read_responses()


class TestMCPClientDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_terminates_process(self):
        client = MCPClient(_stdio_config())
        client._connected = True
        mock_proc = MagicMock()
        client._process = mock_proc
        client._read_task = None

        await client.disconnect()

        mock_proc.terminate.assert_called_once()
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_closes_http_client(self):
        client = MCPClient(_sse_config())
        client._connected = True
        client._process = None
        client._read_task = None
        mock_http = AsyncMock()
        client._http_client = mock_http

        await client.disconnect()

        mock_http.aclose.assert_awaited_once()


# ── MCPServer Tests ──────────────────────────────────────────


class TestMCPServerGetTools:
    def test_no_router_no_executor_returns_empty(self):
        server = MCPServer()
        assert server.get_tools() == []

    def test_with_skill_router_returns_skill_tools(self):
        router = MagicMock()
        router.list_skills.return_value = [
            {"name": "code_executor", "description": "Run code", "parameters": {}},
        ]
        server = MCPServer(skill_router=router)
        tools = server.get_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "code_executor"

    def test_with_tool_executor_returns_builtin_tools(self):
        executor = MagicMock()
        server = MCPServer(tool_executor=executor)
        tools = server.get_tools()
        names = [t["name"] for t in tools]
        assert "shell" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "memory_search" in names


class TestMCPServerHandleRequest:
    @pytest.mark.asyncio
    async def test_handle_tools_list(self):
        router = MagicMock()
        router.list_skills.return_value = [
            {"name": "search", "description": "Search", "parameters": {}},
        ]
        server = MCPServer(skill_router=router)

        resp = await server.handle_request({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}
        })

        assert resp["id"] == 1
        assert "result" in resp
        assert len(resp["result"]["tools"]) == 1

    @pytest.mark.asyncio
    async def test_handle_tools_call_delegates_to_skill(self):
        skill = AsyncMock()
        skill.execute.return_value = {"success": True, "content": "done"}

        router = MagicMock()
        router.get_skill.return_value = skill

        server = MCPServer(skill_router=router)
        resp = await server.handle_request({
            "jsonrpc": "2.0", "id": 2,
            "method": "tools/call",
            "params": {"name": "my_skill", "arguments": {"x": 1}},
        })

        assert resp["result"]["isError"] is False
        assert "done" in resp["result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handle_tools_call_unknown_tool(self):
        router = MagicMock()
        router.get_skill.return_value = None
        server = MCPServer(skill_router=router, tool_executor=None)

        resp = await server.handle_request({
            "jsonrpc": "2.0", "id": 3,
            "method": "tools/call",
            "params": {"name": "nonexistent", "arguments": {}},
        })

        assert resp["result"]["isError"] is True

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self):
        server = MCPServer()
        resp = await server.handle_request({
            "jsonrpc": "2.0", "id": 4, "method": "foo/bar", "params": {}
        })

        assert "error" in resp
        assert resp["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_handle_initialize(self):
        server = MCPServer()
        resp = await server.handle_request({
            "jsonrpc": "2.0", "id": 5, "method": "initialize", "params": {}
        })

        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert "serverInfo" in resp["result"]

    @pytest.mark.asyncio
    async def test_malformed_request_returns_error(self):
        """A request whose handler raises should return a JSON-RPC error."""
        server = MCPServer()

        # Patch _handle_initialize to raise
        with patch.object(server, "_handle_initialize", side_effect=RuntimeError("boom")):
            resp = await server.handle_request({
                "jsonrpc": "2.0", "id": 6, "method": "initialize", "params": {}
            })

        assert "error" in resp
        assert resp["error"]["code"] == -32603
        assert "boom" in resp["error"]["message"]


class TestMCPServerFormatResult:
    def test_success_dict(self):
        server = MCPServer()
        result = server._format_tool_result({"success": True, "content": "ok"})
        assert result["isError"] is False
        assert result["content"][0]["text"] == "ok"

    def test_failure_dict(self):
        server = MCPServer()
        result = server._format_tool_result({"success": False, "error": "bad"})
        assert result["isError"] is True
        assert "bad" in result["content"][0]["text"]

    def test_non_dict_result(self):
        server = MCPServer()
        result = server._format_tool_result("plain string")
        assert result["isError"] is False
        assert result["content"][0]["text"] == "plain string"


class TestMCPServerStop:
    def test_stop_sets_running_false(self):
        server = MCPServer()
        server._running = True
        server.stop()
        assert server._running is False


# ── MCPRegistry Tests ────────────────────────────────────────


class TestMCPRegistryInit:
    def test_starts_empty(self):
        reg = MCPRegistry.__new__(MCPRegistry)
        reg._clients = {}
        reg._tool_map = {}
        reg._initialized = False
        reg._approval_middleware = None

        assert reg.connected_servers == []
        assert reg.total_tools == 0


class TestMCPRegistryInitialize:
    @pytest.mark.asyncio
    async def test_no_servers_configured(self):
        settings = MagicMock()
        settings.get.return_value = {}

        reg = MCPRegistry.__new__(MCPRegistry)
        reg.settings = settings
        reg._clients = {}
        reg._tool_map = {}
        reg._initialized = False
        reg._approval_middleware = None

        await reg.initialize()
        assert reg._initialized is True
        assert reg.connected_servers == []

    @pytest.mark.asyncio
    async def test_connects_to_configured_server(self):
        settings = MagicMock()
        settings.get.return_value = {
            "myserver": {"transport": "stdio", "command": "fake", "enabled": True},
        }

        mock_client = MagicMock(spec=MCPClient)
        mock_client.connect = AsyncMock(return_value=True)
        mock_client.connected = True
        mock_client.get_tools.return_value = SAMPLE_TOOLS
        mock_client.tool_count = 2

        reg = MCPRegistry.__new__(MCPRegistry)
        reg.settings = settings
        reg._clients = {}
        reg._tool_map = {}
        reg._initialized = False
        reg._approval_middleware = None

        with patch("openclaw.mcp.registry.MCPClient", return_value=mock_client):
            await reg.initialize()

        assert "myserver" in reg._clients
        assert reg.total_tools == 2
        # Both short and full names are mapped
        assert "read" in reg._tool_map
        assert "myserver_read" in reg._tool_map

    @pytest.mark.asyncio
    async def test_skips_disabled_server(self):
        settings = MagicMock()
        settings.get.return_value = {
            "disabled": {"transport": "stdio", "command": "x", "enabled": False},
        }

        reg = MCPRegistry.__new__(MCPRegistry)
        reg.settings = settings
        reg._clients = {}
        reg._tool_map = {}
        reg._initialized = False
        reg._approval_middleware = None

        await reg.initialize()
        assert len(reg._clients) == 0

    @pytest.mark.asyncio
    async def test_failed_connection_skipped(self):
        settings = MagicMock()
        settings.get.return_value = {
            "bad": {"transport": "stdio", "command": "x", "enabled": True},
        }

        mock_client = MagicMock(spec=MCPClient)
        mock_client.connect = AsyncMock(return_value=False)

        reg = MCPRegistry.__new__(MCPRegistry)
        reg.settings = settings
        reg._clients = {}
        reg._tool_map = {}
        reg._initialized = False
        reg._approval_middleware = None

        with patch("openclaw.mcp.registry.MCPClient", return_value=mock_client):
            await reg.initialize()

        assert len(reg._clients) == 0


class TestMCPRegistryCallTool:
    @pytest.mark.asyncio
    async def test_call_known_tool(self):
        mock_client = MagicMock()
        mock_client.connected = True
        mock_client.call_tool = AsyncMock(return_value={
            "success": True, "content": "result",
        })

        reg = MCPRegistry.__new__(MCPRegistry)
        reg.settings = MagicMock()
        reg.settings.get.return_value = {}
        reg._clients = {"srv": mock_client}
        reg._tool_map = {"read": "srv", "srv_read": "srv"}
        reg._initialized = True
        reg._approval_middleware = None

        result = await reg.call_tool("read", {"path": "/x"})
        assert result["success"] is True
        mock_client.call_tool.assert_awaited_once_with("read", {"path": "/x"})

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        reg = MCPRegistry.__new__(MCPRegistry)
        reg.settings = MagicMock()
        reg.settings.get.return_value = {}
        reg._clients = {}
        reg._tool_map = {}
        reg._initialized = True
        reg._approval_middleware = None

        result = await reg.call_tool("nonexistent", {})
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    @pytest.mark.asyncio
    async def test_call_tool_strips_server_prefix(self):
        mock_client = MagicMock()
        mock_client.connected = True
        mock_client.call_tool = AsyncMock(return_value={"success": True, "content": "ok"})

        reg = MCPRegistry.__new__(MCPRegistry)
        reg.settings = MagicMock()
        reg.settings.get.return_value = {}
        reg._clients = {"srv": mock_client}
        reg._tool_map = {"srv_read": "srv"}
        reg._initialized = True
        reg._approval_middleware = None

        await reg.call_tool("srv_read", {"path": "/"})
        mock_client.call_tool.assert_awaited_once_with("read", {"path": "/"})

    @pytest.mark.asyncio
    async def test_call_tool_denied_by_approval(self):
        mock_client = MagicMock()
        mock_client.connected = True

        approval = MagicMock()
        approval.check_approval = AsyncMock(return_value=(False, "needs human"))

        reg = MCPRegistry.__new__(MCPRegistry)
        reg.settings = MagicMock()
        reg.settings.get.return_value = {}
        reg._clients = {"srv": mock_client}
        reg._tool_map = {"read": "srv"}
        reg._initialized = True
        reg._approval_middleware = approval

        result = await reg.call_tool("read", {}, session_id="s1")
        assert result["success"] is False
        assert result["approval_required"] is True

    @pytest.mark.asyncio
    async def test_call_tool_disconnected_server(self):
        mock_client = MagicMock()
        mock_client.connected = False

        reg = MCPRegistry.__new__(MCPRegistry)
        reg.settings = MagicMock()
        reg.settings.get.return_value = {}
        reg._clients = {"srv": mock_client}
        reg._tool_map = {"read": "srv"}
        reg._initialized = True
        reg._approval_middleware = None

        result = await reg.call_tool("read", {})
        assert result["success"] is False
        assert "not connected" in result["error"]


class TestMCPRegistryGetTools:
    def test_get_all_tools(self):
        mock_client = MagicMock()
        mock_client.get_tools.return_value = SAMPLE_TOOLS

        reg = MCPRegistry.__new__(MCPRegistry)
        reg._clients = {"srv": mock_client}

        tools = reg.get_all_tools()
        assert len(tools) == 2
        assert tools[0]["server"] == "srv"
        assert tools[0]["full_name"] == "srv_read"

    def test_get_tools_for_llm(self):
        mock_client = MagicMock()
        mock_client.get_tools_for_llm.return_value = [
            {"type": "function", "function": {"name": "srv_read"}},
        ]

        reg = MCPRegistry.__new__(MCPRegistry)
        reg._clients = {"srv": mock_client}

        tools = reg.get_tools_for_llm()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"


class TestMCPRegistryDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_all(self):
        c1 = MagicMock()
        c1.disconnect = AsyncMock()
        c2 = MagicMock()
        c2.disconnect = AsyncMock()

        reg = MCPRegistry.__new__(MCPRegistry)
        reg._clients = {"a": c1, "b": c2}
        reg._tool_map = {"tool1": "a"}
        reg._initialized = True

        await reg.disconnect_all()

        c1.disconnect.assert_awaited_once()
        c2.disconnect.assert_awaited_once()
        assert len(reg._clients) == 0
        assert len(reg._tool_map) == 0
        assert reg._initialized is False
