"""
MCP Registry - Manages multiple MCP server connections.
Provides unified access to tools from all connected MCP servers.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from openclaw.config.settings import get_settings
from openclaw.mcp.client import MCPClient

logger = logging.getLogger("openclaw.mcp.registry")


class MCPRegistry:
    """
    Registry for managing multiple MCP server connections.

    Features:
    - Auto-discovery from config
    - Connection pooling
    - Tool routing
    - Graceful reconnection
    """

    def __init__(self, approval_middleware=None):
        self.settings = get_settings()
        self._clients: dict[str, MCPClient] = {}
        self._tool_map: dict[str, str] = {}  # tool_name -> server_name
        self._initialized = False
        self._approval_middleware = approval_middleware

    async def initialize(self):
        """Initialize and connect to all configured MCP servers."""
        if self._initialized:
            return

        servers = self.settings.get("mcp.servers", {})
        if not servers:
            logger.info("No MCP servers configured")
            self._initialized = True
            return

        for name, config in servers.items():
            if not config.get("enabled", True):
                continue

            config["name"] = name
            client = MCPClient(config)

            try:
                if await client.connect():
                    self._clients[name] = client
                    # Map tools to this server
                    for tool in client.get_tools():
                        full_name = f"{name}_{tool['name']}"
                        self._tool_map[full_name] = name
                        self._tool_map[tool['name']] = name  # Also allow short name
            except Exception as e:
                logger.error(f"Failed to connect to MCP server {name}: {e}")

        self._initialized = True
        logger.info(f"MCP Registry initialized with {len(self._clients)} servers, {len(self._tool_map)} tools")

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict,
        session_id: str = "",
    ) -> dict:
        """
        Call a tool by name, routing to the appropriate MCP server.
        Integrates with ApprovalMiddleware for sensitive operations.

        Args:
            tool_name: Full or short tool name
            arguments: Tool arguments
            session_id: Session ID for approval context

        Returns:
            Tool result
        """
        await self.initialize()

        # Find the server for this tool
        server_name = self._tool_map.get(tool_name)
        if not server_name:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        client = self._clients.get(server_name)
        if not client or not client.connected:
            return {"success": False, "error": f"MCP server not connected: {server_name}"}

        # Extract the actual tool name (remove server prefix if present)
        actual_name = tool_name
        if tool_name.startswith(f"{server_name}_"):
            actual_name = tool_name[len(server_name) + 1:]

        # Human-in-the-Loop approval check
        if self._approval_middleware:
            approved, reason = await self._approval_middleware.check_approval(
                tool_name=actual_name,
                server_name=server_name,
                arguments=arguments,
                session_id=session_id,
            )
            if not approved:
                logger.warning(
                    f"MCP tool call DENIED: {actual_name} (reason={reason})"
                )
                return {
                    "success": False,
                    "error": f"Tool call denied: {reason}",
                    "approval_required": True,
                }

        return await client.call_tool(actual_name, arguments)

    def get_all_tools(self) -> list[dict]:
        """Get all tools from all connected servers."""
        tools = []
        for name, client in self._clients.items():
            for tool in client.get_tools():
                tools.append({
                    "server": name,
                    "name": tool.get("name"),
                    "full_name": f"{name}_{tool.get('name')}",
                    "description": tool.get("description", ""),
                    "inputSchema": tool.get("inputSchema", {}),
                })
        return tools

    def get_tools_for_llm(self) -> list[dict]:
        """Get all tools formatted for LLM function calling."""
        tools = []
        for client in self._clients.values():
            tools.extend(client.get_tools_for_llm())
        return tools

    def get_tools_description(self) -> str:
        """Get a human-readable description of all available MCP tools."""
        lines = ["### MCP Tools (External Integrations)\n"]

        for name, client in self._clients.items():
            tools = client.get_tools()
            if tools:
                lines.append(f"\n**{name}** ({len(tools)} tools):")
                for tool in tools:
                    desc = tool.get("description", "No description")[:80]
                    lines.append(f"  - `{tool['name']}`: {desc}")

        return "\n".join(lines)

    async def disconnect_all(self):
        """Disconnect from all MCP servers."""
        for client in self._clients.values():
            try:
                await client.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting from {client.name}: {e}")

        self._clients.clear()
        self._tool_map.clear()
        self._initialized = False

    @property
    def connected_servers(self) -> list[str]:
        return [name for name, client in self._clients.items() if client.connected]

    @property
    def total_tools(self) -> int:
        return sum(client.tool_count for client in self._clients.values())


# Singleton instance
_registry: Optional[MCPRegistry] = None


def get_mcp_registry(approval_middleware=None) -> MCPRegistry:
    """Get the global MCP registry instance."""
    global _registry
    if _registry is None:
        _registry = MCPRegistry(approval_middleware=approval_middleware)
    elif approval_middleware is not None and _registry._approval_middleware is None:
        _registry._approval_middleware = approval_middleware
    return _registry
