"""
MCP Client - Connect to MCP servers and invoke their tools.
Implements the Model Context Protocol client specification.
"""

import asyncio
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("openclaw.mcp.client")


class MCPClient:
    """
    Client for connecting to MCP (Model Context Protocol) servers.

    Supports:
    - stdio transport (subprocess)
    - SSE transport (HTTP)
    - Tool discovery
    - Tool invocation
    - Resource access
    """

    def __init__(self, server_config: dict):
        """
        Initialize MCP client.

        Args:
            server_config: Configuration for the MCP server
                - command: Command to start the server (for stdio)
                - args: Arguments for the command
                - env: Environment variables
                - url: URL for SSE transport
        """
        self.config = server_config
        self.name = server_config.get("name", "mcp-server")
        self._process: Optional[subprocess.Popen] = None
        self._tools: list[dict] = []
        self._resources: list[dict] = []
        self._prompts: list[dict] = []
        self._connected = False
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self._read_task: Optional[asyncio.Task] = None

    async def connect(self) -> bool:
        """Connect to the MCP server."""
        transport = self.config.get("transport", "stdio")

        try:
            if transport == "stdio":
                await self._connect_stdio()
            elif transport == "sse":
                await self._connect_sse()
            else:
                raise ValueError(f"Unknown transport: {transport}")

            # Initialize the connection
            await self._initialize()
            self._connected = True
            logger.info(f"Connected to MCP server: {self.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.name}: {e}")
            return False

    async def _connect_stdio(self):
        """Connect via stdio (subprocess)."""
        command = self.config.get("command")
        args = self.config.get("args", [])
        env = self.config.get("env", {})

        if not command:
            raise ValueError("No command specified for stdio transport")

        # Merge environment
        import os
        full_env = {**os.environ, **env}

        # Start the subprocess
        self._process = subprocess.Popen(
            [command] + args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=full_env,
            bufsize=0,
        )

        # Start reading responses
        self._read_task = asyncio.create_task(self._read_responses())

    async def _connect_sse(self):
        """Connect via SSE (HTTP Server-Sent Events)."""
        # SSE transport implementation
        url = self.config.get("url")
        if not url:
            raise ValueError("No URL specified for SSE transport")

        # For SSE, we'll use httpx with streaming
        import httpx
        self._http_client = httpx.AsyncClient(timeout=30)
        self._sse_url = url

    async def _initialize(self):
        """Send initialize request to the server."""
        response = await self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {"listChanged": True},
                "sampling": {},
            },
            "clientInfo": {
                "name": "OpenClaw",
                "version": "1.0.0",
            }
        })

        if response.get("error"):
            raise Exception(f"Initialize failed: {response['error']}")

        # Send initialized notification
        await self._notify("notifications/initialized", {})

        # List available tools
        await self._discover_tools()
        await self._discover_resources()

    async def _discover_tools(self):
        """Discover available tools from the server."""
        response = await self._request("tools/list", {})
        self._tools = response.get("result", {}).get("tools", [])
        logger.info(f"Discovered {len(self._tools)} tools from {self.name}")

    async def _discover_resources(self):
        """Discover available resources from the server."""
        try:
            response = await self._request("resources/list", {})
            self._resources = response.get("result", {}).get("resources", [])
            logger.info(f"Discovered {len(self._resources)} resources from {self.name}")
        except Exception:
            # Resources are optional
            pass

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if not self._connected:
            raise RuntimeError("Not connected to MCP server")

        response = await self._request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })

        if response.get("error"):
            return {"success": False, "error": response["error"]}

        result = response.get("result", {})
        content = result.get("content", [])

        # Extract text content
        text_parts = []
        for item in content:
            if item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        return {
            "success": True,
            "content": "\n".join(text_parts),
            "raw": content,
            "isError": result.get("isError", False),
        }

    async def read_resource(self, uri: str) -> dict:
        """Read a resource from the MCP server."""
        if not self._connected:
            raise RuntimeError("Not connected to MCP server")

        response = await self._request("resources/read", {"uri": uri})

        if response.get("error"):
            return {"success": False, "error": response["error"]}

        return {
            "success": True,
            "contents": response.get("result", {}).get("contents", []),
        }

    def get_tools(self) -> list[dict]:
        """Get list of available tools."""
        return self._tools

    def get_tools_for_llm(self) -> list[dict]:
        """Get tools formatted for LLM function calling."""
        formatted = []
        for tool in self._tools:
            formatted.append({
                "type": "function",
                "function": {
                    "name": f"{self.name}_{tool['name']}",
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {}),
                }
            })
        return formatted

    async def _request(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC request and wait for response."""
        self._request_id += 1
        request_id = self._request_id

        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        future = asyncio.get_event_loop().create_future()
        self._pending_requests[request_id] = future

        await self._send(message)

        try:
            response = await asyncio.wait_for(future, timeout=30)
            return response
        except asyncio.TimeoutError:
            del self._pending_requests[request_id]
            raise TimeoutError(f"Request {method} timed out")

    async def _notify(self, method: str, params: dict):
        """Send a JSON-RPC notification (no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        await self._send(message)

    async def _send(self, message: dict):
        """Send a message to the server."""
        if self._process and self._process.stdin:
            data = json.dumps(message) + "\n"
            self._process.stdin.write(data.encode())
            self._process.stdin.flush()
        elif hasattr(self, "_http_client"):
            # SSE transport - use HTTP POST
            await self._http_client.post(
                f"{self._sse_url}/message",
                json=message
            )

    async def _read_responses(self):
        """Read responses from the server (stdio)."""
        while self._process and self._process.stdout:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, self._process.stdout.readline
                )
                if not line:
                    break

                message = json.loads(line.decode())
                request_id = message.get("id")

                if request_id and request_id in self._pending_requests:
                    future = self._pending_requests.pop(request_id)
                    if not future.done():
                        future.set_result(message)

            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading MCP response: {e}")
                break

    async def disconnect(self):
        """Disconnect from the MCP server."""
        self._connected = False

        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass

        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()

        if hasattr(self, "_http_client"):
            await self._http_client.aclose()

        logger.info(f"Disconnected from MCP server: {self.name}")

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tool_count(self) -> int:
        return len(self._tools)
