"""
MCP Human-in-the-Loop Approval Middleware.

Classifies MCP tools as "safe" (read) or "sensitive" (write/delete),
and pauses execution for sensitive operations until the user approves
via WebSocket notification.

Flow:
1. Agent calls an MCP tool
2. Middleware checks tool classification
3. If "safe" -> execute immediately
4. If "sensitive" -> pause, notify UI via WebSocket, wait for approval
5. On approval -> execute; on denial -> return denied result
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

from openclaw.config.settings import get_settings

if TYPE_CHECKING:
    from openclaw.gateway.server import ConnectionManager

logger = logging.getLogger("openclaw.gateway.approval")


class ToolSafety(str, Enum):
    """Safety classification for MCP tools."""
    SAFE = "safe"             # Read-only, no side effects
    SENSITIVE = "sensitive"   # Write, delete, or destructive actions
    CRITICAL = "critical"     # Irreversible or high-impact actions


# Default classification rules based on tool name patterns
SAFETY_RULES = {
    # Patterns that indicate safe (read-only) operations
    "safe_patterns": [
        "get_", "list_", "read_", "search_", "find_",
        "describe_", "show_", "view_", "fetch_", "count_",
        "check_", "status_", "info_", "stat_", "head_",
    ],
    # Patterns that indicate sensitive (write) operations
    "sensitive_patterns": [
        "write_", "create_", "update_", "edit_", "modify_",
        "add_", "set_", "put_", "post_", "upload_",
        "push_", "commit_", "merge_", "send_",
    ],
    # Patterns that indicate critical (destructive) operations
    "critical_patterns": [
        "delete_", "remove_", "destroy_", "drop_", "purge_",
        "force_", "reset_", "revoke_", "terminate_", "kill_",
    ],
}

# Explicit overrides for known MCP tools
TOOL_OVERRIDES: dict[str, ToolSafety] = {
    # GitHub
    "get_file_contents": ToolSafety.SAFE,
    "list_repos": ToolSafety.SAFE,
    "search_code": ToolSafety.SAFE,
    "create_issue": ToolSafety.SENSITIVE,
    "create_pull_request": ToolSafety.SENSITIVE,
    "push_files": ToolSafety.SENSITIVE,
    "delete_repo": ToolSafety.CRITICAL,
    "delete_branch": ToolSafety.CRITICAL,
    # Filesystem
    "read_file": ToolSafety.SAFE,
    "list_directory": ToolSafety.SAFE,
    "write_file": ToolSafety.SENSITIVE,
    "delete_file": ToolSafety.CRITICAL,
    # Slack
    "list_channels": ToolSafety.SAFE,
    "send_message": ToolSafety.SENSITIVE,
    "delete_message": ToolSafety.CRITICAL,
}


@dataclass
class ApprovalRequest:
    """A pending approval request waiting for user decision."""
    id: str = field(default_factory=lambda: f"approval_{uuid.uuid4().hex[:8]}")
    tool_name: str = ""
    server_name: str = ""
    arguments: dict = field(default_factory=dict)
    safety_level: str = ToolSafety.SENSITIVE
    description: str = ""
    session_id: str = ""
    created_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending, approved, denied, expired
    decided_at: Optional[float] = None
    decided_by: str = ""
    _future: Optional[asyncio.Future] = field(default=None, repr=False)


class ApprovalMiddleware:
    """
    Middleware that intercepts sensitive MCP tool calls and requires
    human approval before execution.

    Integrates with the WebSocket ConnectionManager to send real-time
    approval notifications to the UI.
    """

    def __init__(self, ws_manager: "ConnectionManager" = None):
        self.settings = get_settings()
        self._enabled = self.settings.get("mcp.approval.enabled", True)
        self._timeout = self.settings.get("mcp.approval.timeout_seconds", 120)
        self._auto_approve_safe = self.settings.get("mcp.approval.auto_approve_safe", True)
        self._ws_manager = ws_manager
        self._pending: dict[str, ApprovalRequest] = {}
        self._max_history = 500
        self._history: deque = deque(maxlen=self._max_history)
        self._custom_overrides: dict[str, ToolSafety] = {}

        # Load custom overrides from config
        config_overrides = self.settings.get("mcp.approval.tool_overrides", {})
        for tool, level in config_overrides.items():
            if level in (ToolSafety.SAFE, ToolSafety.SENSITIVE, ToolSafety.CRITICAL):
                self._custom_overrides[tool] = ToolSafety(level)

    def classify_tool(self, tool_name: str, server_name: str = "") -> ToolSafety:
        """
        Classify an MCP tool's safety level.

        Priority:
        1. Custom overrides from config
        2. Explicit tool overrides (TOOL_OVERRIDES)
        3. Pattern matching on tool name
        4. Default to SENSITIVE (safe by default = unsafe)
        """
        full_name = f"{server_name}_{tool_name}" if server_name else tool_name

        # Custom overrides (highest priority)
        if full_name in self._custom_overrides:
            return self._custom_overrides[full_name]
        if tool_name in self._custom_overrides:
            return self._custom_overrides[tool_name]

        # Explicit overrides
        if tool_name in TOOL_OVERRIDES:
            return TOOL_OVERRIDES[tool_name]

        # Pattern matching
        tool_lower = tool_name.lower()

        for pattern in SAFETY_RULES["critical_patterns"]:
            if tool_lower.startswith(pattern) or pattern in tool_lower:
                return ToolSafety.CRITICAL

        for pattern in SAFETY_RULES["sensitive_patterns"]:
            if tool_lower.startswith(pattern) or pattern in tool_lower:
                return ToolSafety.SENSITIVE

        for pattern in SAFETY_RULES["safe_patterns"]:
            if tool_lower.startswith(pattern) or pattern in tool_lower:
                return ToolSafety.SAFE

        # Default: treat unknown as sensitive
        return ToolSafety.SENSITIVE

    async def check_approval(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict,
        session_id: str = "",
    ) -> tuple[bool, str]:
        """
        Check if a tool call needs approval and handle the flow.

        Returns:
            (approved: bool, reason: str)
        """
        if not self._enabled:
            return True, "approval_disabled"

        safety = self.classify_tool(tool_name, server_name)

        # Safe tools auto-approved
        if safety == ToolSafety.SAFE and self._auto_approve_safe:
            logger.debug(f"Tool '{tool_name}' auto-approved (safe)")
            return True, "auto_approved_safe"

        # Sensitive/Critical tools need approval
        return await self._request_approval(
            tool_name=tool_name,
            server_name=server_name,
            arguments=arguments,
            safety_level=safety,
            session_id=session_id,
        )

    async def _request_approval(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict,
        safety_level: ToolSafety,
        session_id: str,
    ) -> tuple[bool, str]:
        """Create an approval request and wait for user decision."""
        # Build human-readable description
        description = self._build_description(tool_name, server_name, arguments, safety_level)

        request = ApprovalRequest(
            tool_name=tool_name,
            server_name=server_name,
            arguments=arguments,
            safety_level=safety_level,
            description=description,
            session_id=session_id,
        )

        loop = asyncio.get_running_loop()
        request._future = loop.create_future()
        self._pending[request.id] = request

        logger.info(
            f"Approval requested: {request.id} - {tool_name} "
            f"(level={safety_level}, server={server_name})"
        )

        # Notify UI via WebSocket
        await self._notify_ui(request)

        # Wait for decision with timeout
        try:
            approved = await asyncio.wait_for(
                request._future,
                timeout=self._timeout,
            )
            reason = "user_approved" if approved else "user_denied"
        except asyncio.TimeoutError:
            approved = False
            reason = "timeout"
            request.status = "expired"
            logger.warning(f"Approval {request.id} expired after {self._timeout}s")

        # Cleanup
        self._pending.pop(request.id, None)
        self._history.append({
            "id": request.id,
            "tool": tool_name,
            "server": server_name,
            "safety": safety_level,
            "approved": approved,
            "reason": reason,
            "created_at": request.created_at,
            "decided_at": time.time(),
        })

        return approved, reason

    async def _notify_ui(self, request: ApprovalRequest):
        """Send approval notification to connected UI clients via WebSocket."""
        if not self._ws_manager:
            logger.warning("No WebSocket manager - approval requires manual API call")
            return

        notification = {
            "type": "approval_request",
            "id": request.id,
            "tool_name": request.tool_name,
            "server_name": request.server_name,
            "safety_level": request.safety_level,
            "description": request.description,
            "arguments_preview": self._safe_preview(request.arguments),
            "session_id": request.session_id,
            "created_at": request.created_at,
            "timeout_seconds": self._timeout,
        }

        await self._ws_manager.broadcast(notification)

    def resolve_approval(self, approval_id: str, approved: bool, decided_by: str = "user"):
        """Resolve a pending approval request (called from API/WebSocket)."""
        request = self._pending.get(approval_id)
        if not request:
            logger.warning(f"Approval {approval_id} not found or already resolved")
            return False

        request.status = "approved" if approved else "denied"
        request.decided_at = time.time()
        request.decided_by = decided_by

        if request._future and not request._future.done():
            request._future.set_result(approved)

        logger.info(
            f"Approval {approval_id} resolved: "
            f"{'APPROVED' if approved else 'DENIED'} by {decided_by}"
        )
        return True

    def _build_description(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict,
        safety_level: ToolSafety,
    ) -> str:
        """Build a human-readable description of the action."""
        level_emoji = {
            ToolSafety.SAFE: "[SAFE]",
            ToolSafety.SENSITIVE: "[SENSITIVE]",
            ToolSafety.CRITICAL: "[CRITICAL]",
        }
        prefix = level_emoji.get(safety_level, "[UNKNOWN]")
        args_preview = ", ".join(
            f"{k}={repr(v)[:50]}" for k, v in list(arguments.items())[:5]
        )
        return (
            f"{prefix} L'agent veut executer '{tool_name}' via {server_name} MCP. "
            f"Arguments: {args_preview}"
        )

    @staticmethod
    def _safe_preview(arguments: dict, max_len: int = 200) -> dict:
        """Create a safe preview of arguments (truncated, no secrets)."""
        preview = {}
        for key, value in arguments.items():
            str_val = str(value)
            if len(str_val) > max_len:
                str_val = str_val[:max_len] + "..."
            # Mask potential secrets
            key_lower = key.lower()
            if any(s in key_lower for s in ("token", "secret", "password", "key", "auth")):
                str_val = "***REDACTED***"
            preview[key] = str_val
        return preview

    def get_pending(self) -> list[dict]:
        """List all pending approval requests."""
        return [
            {
                "id": r.id,
                "tool": r.tool_name,
                "server": r.server_name,
                "safety": r.safety_level,
                "description": r.description,
                "created_at": r.created_at,
                "timeout_seconds": self._timeout,
            }
            for r in self._pending.values()
        ]

    def get_history(self, limit: int = 50) -> list[dict]:
        """Get recent approval history."""
        return list(self._history)[-limit:]
