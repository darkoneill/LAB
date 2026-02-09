"""
MCP Server - Expose OpenClaw skills as MCP tools.
Allows other MCP clients to connect and use OpenClaw's capabilities.
"""

import asyncio
import json
import logging
import sys
from typing import Optional

from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.mcp.server")


class MCPServer:
    """
    MCP Server that exposes OpenClaw skills as MCP-compatible tools.

    Supports stdio transport for integration with Claude Desktop, etc.
    """

    def __init__(self, skill_router=None, tool_executor=None):
        self.settings = get_settings()
        self.skill_router = skill_router
        self.tool_executor = tool_executor
        self._running = False

    def get_server_info(self) -> dict:
        """Get server capabilities."""
        return {
            "name": "openclaw",
            "version": self.settings.get("app.version", "1.0.0"),
        }

    def get_tools(self) -> list[dict]:
        """Get list of available tools (from skills)."""
        tools = []

        # Add skills as tools
        if self.skill_router:
            for skill in self.skill_router.list_skills():
                tools.append({
                    "name": skill.get("name", ""),
                    "description": skill.get("description", ""),
                    "inputSchema": {
                        "type": "object",
                        "properties": skill.get("parameters", {}),
                    }
                })

        # Add built-in tools
        if self.tool_executor:
            builtin_tools = [
                {
                    "name": "shell",
                    "description": "Execute a shell command",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The shell command to execute"
                            }
                        },
                        "required": ["command"]
                    }
                },
                {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file"
                            }
                        },
                        "required": ["path"]
                    }
                },
                {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write"
                            }
                        },
                        "required": ["path", "content"]
                    }
                },
                {
                    "name": "memory_search",
                    "description": "Search OpenClaw's memory for relevant information",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            ]
            tools.extend(builtin_tools)

        return tools

    async def handle_request(self, request: dict) -> dict:
        """Handle an incoming JSON-RPC request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = {"tools": self.get_tools()}
            elif method == "tools/call":
                result = await self._handle_tool_call(params)
            elif method == "resources/list":
                result = {"resources": []}
            elif method == "prompts/list":
                result = {"prompts": []}
            else:
                return self._error_response(request_id, -32601, f"Unknown method: {method}")

            return self._success_response(request_id, result)

        except Exception as e:
            logger.error(f"Error handling {method}: {e}")
            return self._error_response(request_id, -32603, str(e))

    async def _handle_initialize(self, params: dict) -> dict:
        """Handle initialize request."""
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
                "prompts": {"listChanged": False},
            },
            "serverInfo": self.get_server_info(),
        }

    async def _handle_tool_call(self, params: dict) -> dict:
        """Handle tool/call request."""
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # Try skill first
        if self.skill_router:
            skill = self.skill_router.get_skill(tool_name)
            if skill:
                result = await skill.execute(**arguments)
                return self._format_tool_result(result)

        # Try tool executor
        if self.tool_executor:
            result = await self.tool_executor.execute(tool_name, arguments)
            return self._format_tool_result(result)

        return {
            "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
            "isError": True,
        }

    def _format_tool_result(self, result: dict) -> dict:
        """Format a tool result for MCP response."""
        if isinstance(result, dict):
            if result.get("success", True):
                content = result.get("content") or result.get("result") or json.dumps(result)
                if isinstance(content, dict):
                    content = json.dumps(content, indent=2)
                return {
                    "content": [{"type": "text", "text": str(content)}],
                    "isError": False,
                }
            else:
                return {
                    "content": [{"type": "text", "text": result.get("error", "Unknown error")}],
                    "isError": True,
                }
        else:
            return {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False,
            }

    def _success_response(self, request_id, result: dict) -> dict:
        """Create a success response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    def _error_response(self, request_id, code: int, message: str) -> dict:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message,
            }
        }

    async def run_stdio(self):
        """Run the server using stdio transport."""
        self._running = True
        logger.info("Starting MCP server (stdio)")

        while self._running:
            try:
                # Read line from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )

                if not line:
                    break

                # Parse JSON-RPC request
                try:
                    request = json.loads(line.strip())
                except json.JSONDecodeError:
                    continue

                # Handle notifications (no id)
                if "id" not in request:
                    # Just log and continue
                    logger.debug(f"Received notification: {request.get('method')}")
                    continue

                # Handle request
                response = await self.handle_request(request)

                # Write response
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()

            except Exception as e:
                logger.error(f"Error in MCP server loop: {e}")

        logger.info("MCP server stopped")

    def stop(self):
        """Stop the server."""
        self._running = False
