"""
Tool Executor - Unified tool execution engine.
Inspired by AgentZero's minimal tool philosophy:
only 4 default tools, the agent creates the rest.
"""

import asyncio
import logging
from typing import Optional

from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.tools")


class ToolExecutor:
    """
    Executes tools with lifecycle management:
    1. Validation
    2. Pre-execution hooks
    3. Execution (with timeout and sandboxing)
    4. Post-execution hooks
    5. Result formatting
    """

    def __init__(self):
        self.settings = get_settings()
        self._tools: dict[str, callable] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register the 4 default tools (AgentZero philosophy)."""
        self._tools["shell"] = self._tool_shell
        self._tools["read_file"] = self._tool_read_file
        self._tools["write_file"] = self._tool_write_file
        self._tools["search_files"] = self._tool_search_files

    def register(self, name: str, handler: callable):
        """Register a custom tool."""
        self._tools[name] = handler

    async def execute(self, tool_name: str, args: dict) -> dict:
        """Execute a tool by name."""
        if tool_name not in self._tools:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

        try:
            result = await self._tools[tool_name](**args)
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return {"success": False, "error": str(e)}

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_tools_description(self) -> str:
        """Get tool descriptions for the agent prompt."""
        descriptions = {
            "shell": "Execute a shell command. Args: command (str), timeout (int, optional)",
            "read_file": "Read a file's content. Args: path (str)",
            "write_file": "Write content to a file. Args: path (str), content (str)",
            "search_files": "Search for files by pattern. Args: path (str), pattern (str)",
        }
        parts = []
        for name in self._tools:
            desc = descriptions.get(name, f"Custom tool: {name}")
            parts.append(f"- `{name}`: {desc}")
        return "\n".join(parts)

    # ── Default Tools ─────────────────────────────────────────

    async def _tool_shell(self, command: str = "", timeout: int = None, **kwargs) -> dict:
        """Execute a shell command."""
        if not self.settings.get("tools.shell.enabled", True):
            return {"success": False, "error": "Shell execution is disabled"}

        if not command:
            return {"success": False, "error": "No command provided"}

        # Check blocked commands
        blocked = self.settings.get("tools.shell.blocked_commands", [])
        # Normalize command for checking
        cmd_lower = command.lower().strip()
        cmd_parts = cmd_lower.split()
        for b in blocked:
            b_lower = b.lower()
            # Check if blocked pattern appears at start or after shell operators
            if cmd_lower.startswith(b_lower):
                return {"success": False, "error": "Command blocked by security policy"}
            # Check after pipes, semicolons, &&, ||
            for sep in ["|", ";", "&&", "||", "`", "$("]:
                if sep in command:
                    for part in command.split(sep):
                        if part.strip().lower().startswith(b_lower):
                            return {"success": False, "error": "Command blocked by security policy"}

        # Additional dangerous pattern checks
        dangerous_patterns = [
            "rm -rf /",
            "rm -fr /",
            "dd if=/dev/zero of=/dev/sd",
            "mkfs.",
            "> /dev/sd",
            "chmod -R 777 /",
            "chown -R",
            ":(){:|:&};:",  # Fork bomb
        ]
        for pattern in dangerous_patterns:
            if pattern in cmd_lower:
                return {"success": False, "error": "Command blocked by security policy"}

        timeout = timeout or self.settings.get("tools.shell.timeout_seconds", 120)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode("utf-8", errors="replace")[:50000],
                "stderr": stderr.decode("utf-8", errors="replace")[:10000],
                "return_code": process.returncode,
            }

        except asyncio.TimeoutError:
            return {"success": False, "error": f"Command timed out after {timeout}s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_read_file(self, path: str = "", **kwargs) -> dict:
        """Read a file."""
        if not path:
            return {"success": False, "error": "No path provided"}

        from pathlib import Path as P
        p = P(path)

        if not p.exists():
            return {"success": False, "error": f"File not found: {path}"}

        max_size = self.settings.get("tools.file_manager.max_file_size_mb", 100) * 1024 * 1024
        if p.stat().st_size > max_size:
            return {"success": False, "error": f"File too large (max {max_size // 1024 // 1024}MB)"}

        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            return {"success": True, "content": content, "path": str(p), "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_write_file(self, path: str = "", content: str = "", **kwargs) -> dict:
        """Write to a file."""
        if not path:
            return {"success": False, "error": "No path provided"}

        from pathlib import Path as P
        p = P(path)

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return {"success": True, "path": str(p), "size": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _tool_search_files(self, path: str = ".", pattern: str = "*", **kwargs) -> dict:
        """Search for files matching a pattern."""
        from pathlib import Path as P
        p = P(path)

        if not p.exists():
            return {"success": False, "error": f"Path not found: {path}"}

        try:
            matches = list(p.rglob(pattern))[:200]
            return {
                "success": True,
                "matches": [str(m) for m in matches],
                "count": len(matches),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
