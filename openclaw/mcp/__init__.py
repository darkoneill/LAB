"""
MCP (Model Context Protocol) Support
Enables OpenClaw to connect to MCP servers and use external tools.
"""

from openclaw.mcp.client import MCPClient
from openclaw.mcp.server import MCPServer
from openclaw.mcp.registry import MCPRegistry

__all__ = ["MCPClient", "MCPServer", "MCPRegistry"]
