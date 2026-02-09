"""
Sandbox Executor - Wraps tool execution with sandbox isolation.
Automatically routes dangerous operations to isolated containers.
"""

import asyncio
import logging
import re
from typing import Optional

from openclaw.config.settings import get_settings
from openclaw.sandbox.container import ContainerManager

logger = logging.getLogger("openclaw.sandbox.executor")


# Patterns that indicate dangerous commands needing sandboxing
DANGEROUS_PATTERNS = [
    r"rm\s+(-[rRf]+\s+)?/",  # rm with root paths
    r"rm\s+.*\*",  # rm with wildcards
    r"dd\s+",  # dd command
    r"mkfs",  # filesystem creation
    r">\s*/dev/",  # writing to devices
    r"chmod\s+(-R\s+)?[0-7]{3}\s+/",  # chmod on root
    r"chown\s+",  # ownership changes
    r"wget\s+.*\|\s*(sh|bash)",  # wget piped to shell
    r"curl\s+.*\|\s*(sh|bash)",  # curl piped to shell
    r"eval\s+",  # eval command
    r"\$\(.*\)",  # command substitution
    r"`.*`",  # backtick command substitution
    r"python.*-c\s+",  # python inline code
    r"perl.*-e\s+",  # perl inline code
    r"ruby.*-e\s+",  # ruby inline code
    r"node.*-e\s+",  # node inline code
]

# Commands that are always safe (don't need sandboxing)
SAFE_COMMANDS = [
    "ls", "pwd", "whoami", "date", "echo", "cat", "head", "tail",
    "grep", "find", "which", "type", "file", "wc", "sort", "uniq",
    "git status", "git log", "git diff", "git branch",
    "docker ps", "docker images",
]


class SandboxExecutor:
    """
    Executor that automatically sandboxes dangerous operations.

    Features:
    - Automatic danger detection
    - Transparent sandboxing
    - Session-based container reuse
    - Graceful fallback
    """

    def __init__(self, base_executor=None):
        self.settings = get_settings()
        self.base_executor = base_executor
        self.container_manager = ContainerManager()
        self._enabled = self.settings.get("sandbox.enabled", True)
        self._force_sandbox = self.settings.get("sandbox.force_all", False)

    async def execute(self, tool_name: str, args: dict, session_id: str = None) -> dict:
        """
        Execute a tool, sandboxing if necessary.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            session_id: Session ID for container reuse

        Returns:
            Execution result
        """
        if not self._enabled:
            return await self._execute_direct(tool_name, args)

        # Check if sandboxing is needed
        needs_sandbox = self._force_sandbox or self._needs_sandbox(tool_name, args)

        if needs_sandbox:
            return await self._execute_sandboxed(tool_name, args, session_id)
        else:
            return await self._execute_direct(tool_name, args)

    def _needs_sandbox(self, tool_name: str, args: dict) -> bool:
        """Determine if execution needs sandboxing."""
        # Shell commands need careful checking
        if tool_name == "shell":
            command = args.get("command", "")
            return self._is_dangerous_command(command)

        # Python code execution always sandboxed
        if tool_name in ("python", "code", "execute_code"):
            return True

        # File write operations to sensitive paths
        if tool_name in ("write_file", "write"):
            path = args.get("path", "")
            return self._is_sensitive_path(path)

        return False

    def _is_dangerous_command(self, command: str) -> bool:
        """Check if a shell command is potentially dangerous."""
        command_lower = command.lower().strip()

        # Check for safe commands
        for safe in SAFE_COMMANDS:
            if command_lower.startswith(safe):
                return False

        # Check dangerous patterns
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                logger.warning(f"Dangerous pattern detected: {pattern}")
                return True

        # Check for pipe to shell
        if re.search(r"\|\s*(bash|sh|zsh)\b", command):
            return True

        # Check for network + execution
        if any(net in command_lower for net in ["wget", "curl", "nc", "netcat"]):
            if any(exec_cmd in command_lower for exec_cmd in ["|", ";", "&&", "$(", "`"]):
                return True

        return False

    def _is_sensitive_path(self, path: str) -> bool:
        """Check if a path is sensitive and should be sandboxed."""
        sensitive_prefixes = [
            "/etc/", "/bin/", "/sbin/", "/usr/bin/", "/usr/sbin/",
            "/boot/", "/root/", "/sys/", "/proc/", "/dev/",
            "~/.ssh/", "~/.bashrc", "~/.profile",
        ]

        for prefix in sensitive_prefixes:
            if path.startswith(prefix):
                return True

        return False

    async def _execute_direct(self, tool_name: str, args: dict) -> dict:
        """Execute directly without sandboxing."""
        if self.base_executor:
            return await self.base_executor.execute(tool_name, args)

        return {
            "success": False,
            "error": "No base executor configured",
        }

    async def _execute_sandboxed(self, tool_name: str, args: dict, session_id: str = None) -> dict:
        """Execute in a sandboxed container."""
        try:
            # Get or create container for session
            container_id = await self.container_manager.get_sandbox_for_session(
                session_id or "default"
            )

            if tool_name == "shell":
                command = args.get("command", "")
                result = await self.container_manager.execute_in_sandbox(
                    container_id,
                    command,
                    timeout=args.get("timeout", 30),
                )
                return {
                    "success": result["success"],
                    "output": result["stdout"],
                    "error": result["stderr"] if not result["success"] else None,
                    "sandboxed": True,
                }

            elif tool_name in ("python", "code", "execute_code"):
                code = args.get("code", args.get("content", ""))
                result = await self.container_manager.execute_python(
                    container_id,
                    code,
                    timeout=args.get("timeout", 30),
                )
                return {
                    "success": result["success"],
                    "output": result["stdout"],
                    "error": result["stderr"] if not result["success"] else None,
                    "sandboxed": True,
                }

            elif tool_name in ("write_file", "write"):
                # Write file in sandbox
                path = args.get("path", "")
                content = args.get("content", "")

                # Map to sandbox path
                sandbox_path = f"/workspace/{path.lstrip('/')}"
                write_cmd = f"mkdir -p $(dirname '{sandbox_path}') && cat > '{sandbox_path}'"

                result = await self.container_manager.execute_in_sandbox(
                    container_id,
                    f"echo '{content}' | {write_cmd}",
                )
                return {
                    "success": result["success"],
                    "path": sandbox_path,
                    "sandboxed": True,
                }

            else:
                # Fallback to direct execution with warning
                logger.warning(f"No sandbox handler for tool: {tool_name}")
                return await self._execute_direct(tool_name, args)

        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return {
                "success": False,
                "error": f"Sandbox error: {str(e)}",
                "sandboxed": True,
            }

    async def cleanup_session(self, session_id: str):
        """Cleanup sandbox for a session."""
        if session_id in self.container_manager._active_containers:
            container_id = self.container_manager._active_containers[session_id]
            await self.container_manager.destroy_sandbox(container_id)

    async def cleanup_all(self):
        """Cleanup all sandboxes."""
        await self.container_manager.cleanup_all()
