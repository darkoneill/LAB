"""
Sandbox Executor - Wraps tool execution with sandbox isolation.
Automatically routes dangerous operations to isolated containers.
Includes Self-Healing Code Loop: auto-corrects code errors via LLM retry.
"""

import base64
import logging
import re
import time
from typing import Optional, TYPE_CHECKING

from openclaw.config.settings import get_settings
from openclaw.sandbox.container import ContainerManager

if TYPE_CHECKING:
    from openclaw.agent.brain import AgentBrain

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

# Error patterns that are good candidates for self-healing
HEALABLE_ERROR_PATTERNS = [
    r"ModuleNotFoundError",
    r"ImportError",
    r"NameError",
    r"SyntaxError",
    r"TypeError",
    r"ValueError",
    r"AttributeError",
    r"KeyError",
    r"IndexError",
    r"FileNotFoundError",
    r"ZeroDivisionError",
    r"IndentationError",
]

SELF_HEALING_PROMPT = """You generated the following Python code that failed with an error.

## Original Code
```python
{code}
```

## Error Output
```
{error}
```

## Attempt {attempt} of {max_attempts}

Analyze the error, understand why it happened, and produce a CORRECTED version of the code.
Rules:
- Return ONLY the corrected Python code, no explanations.
- Do NOT wrap it in markdown code fences.
- If the error is a missing module, try an alternative approach that doesn't require it.
- If the error is a logic bug, fix the logic.
- Preserve the original intent of the code.
"""


class SandboxExecutor:
    """
    Executor that automatically sandboxes dangerous operations.

    Features:
    - Automatic danger detection
    - Transparent sandboxing
    - Session-based container reuse
    - Self-Healing Code Loop (auto-correction via LLM on failure)
    - Graceful fallback
    """

    def __init__(self, base_executor=None, brain: "AgentBrain" = None):
        self.settings = get_settings()
        self.base_executor = base_executor
        self.brain = brain
        self.container_manager = ContainerManager()
        self._enabled = self.settings.get("sandbox.enabled", True)
        self._force_sandbox = self.settings.get("sandbox.force_all", False)
        self._self_healing_enabled = self.settings.get(
            "sandbox.self_healing.enabled", True
        )
        self._max_heal_attempts = self.settings.get(
            "sandbox.self_healing.max_attempts", 3
        )

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
            result = await self._execute_sandboxed(tool_name, args, session_id)

            # Self-Healing: if code execution failed, attempt auto-correction
            if (
                not result.get("success")
                and self._self_healing_enabled
                and self.brain
                and tool_name in ("python", "code", "execute_code")
                and self._is_healable_error(result.get("error", ""))
            ):
                result = await self._self_healing_loop(
                    tool_name, args, session_id, result
                )

            return result
        else:
            return await self._execute_direct(tool_name, args)

    def _is_healable_error(self, error: str) -> bool:
        """Check if an error is a candidate for self-healing."""
        if not error:
            return False
        for pattern in HEALABLE_ERROR_PATTERNS:
            if re.search(pattern, error):
                return True
        return False

    async def _self_healing_loop(
        self,
        tool_name: str,
        original_args: dict,
        session_id: str,
        first_result: dict,
    ) -> dict:
        """
        Self-Healing Code Loop: intercept errors, ask LLM to fix, retry.

        Flow:
        1. Code fails in sandbox with an error
        2. Error + code sent to LLM: "Fix this, here's the error"
        3. LLM returns corrected code
        4. Re-execute in sandbox
        5. Repeat up to max_attempts times

        Returns the last result (success or final failure with healing trace).
        """
        original_code = original_args.get("code", original_args.get("content", ""))
        current_code = original_code
        current_error = first_result.get("error", "")
        healing_trace = []

        healing_trace.append({
            "attempt": 0,
            "type": "original",
            "code": original_code,
            "error": current_error,
            "timestamp": time.time(),
        })

        logger.info(
            f"Self-Healing activated: error='{current_error[:100]}...' "
            f"max_attempts={self._max_heal_attempts}"
        )

        for attempt in range(1, self._max_heal_attempts + 1):
            # Ask the LLM to fix the code
            fix_prompt = SELF_HEALING_PROMPT.format(
                code=current_code,
                error=current_error,
                attempt=attempt,
                max_attempts=self._max_heal_attempts,
            )

            try:
                llm_response = await self.brain.generate(
                    messages=[{"role": "user", "content": fix_prompt}],
                    max_tokens=4096,
                    temperature=0.2,  # Low temperature for precise fixes
                )
                corrected_code = llm_response.get("content", "").strip()
            except Exception as e:
                logger.error(f"Self-Healing: LLM call failed on attempt {attempt}: {e}")
                healing_trace.append({
                    "attempt": attempt,
                    "type": "llm_error",
                    "error": str(e),
                    "timestamp": time.time(),
                })
                break

            if not corrected_code or corrected_code == current_code:
                logger.warning(f"Self-Healing: LLM returned identical or empty code on attempt {attempt}")
                healing_trace.append({
                    "attempt": attempt,
                    "type": "no_change",
                    "timestamp": time.time(),
                })
                break

            # Strip markdown fences if the LLM wrapped it anyway
            corrected_code = self._strip_code_fences(corrected_code)

            # Re-execute the corrected code
            corrected_args = {**original_args}
            if "code" in corrected_args:
                corrected_args["code"] = corrected_code
            else:
                corrected_args["content"] = corrected_code

            result = await self._execute_sandboxed(tool_name, corrected_args, session_id)

            healing_trace.append({
                "attempt": attempt,
                "type": "retry",
                "code": corrected_code,
                "success": result.get("success", False),
                "error": result.get("error") if not result.get("success") else None,
                "output": result.get("output", "")[:500] if result.get("success") else None,
                "timestamp": time.time(),
            })

            if result.get("success"):
                logger.info(f"Self-Healing SUCCESS on attempt {attempt}")
                result["self_healed"] = True
                result["healing_attempts"] = attempt
                result["healing_trace"] = healing_trace
                return result

            # Update for next iteration
            current_code = corrected_code
            current_error = result.get("error", "")
            logger.warning(
                f"Self-Healing attempt {attempt} failed: {current_error[:100]}..."
            )

        # All attempts exhausted
        logger.error(
            f"Self-Healing FAILED after {self._max_heal_attempts} attempts"
        )
        return {
            "success": False,
            "error": (
                f"Self-Healing exhausted ({self._max_heal_attempts} attempts). "
                f"Last error: {current_error}"
            ),
            "self_healed": False,
            "healing_attempts": self._max_heal_attempts,
            "healing_trace": healing_trace,
            "sandboxed": True,
        }

    @staticmethod
    def _strip_code_fences(code: str) -> str:
        """Remove markdown code fences from LLM output."""
        code = code.strip()
        # Handle all common python fence variants
        for prefix in ("```python", "```py", "```"):
            if code.startswith(prefix):
                code = code[len(prefix):]
                break
        if code.endswith("```"):
            code = code[:-3]
        return code.strip()

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
                # Write file in sandbox (safe: base64-encode to avoid injection)
                path = args.get("path", "")
                content = args.get("content", "")

                # Map to sandbox path
                sandbox_path = f"/workspace/{path.lstrip('/')}"
                b64_content = base64.b64encode(content.encode("utf-8")).decode("ascii")
                write_cmd = (
                    f"mkdir -p \"$(dirname '{sandbox_path}')\" && "
                    f"echo '{b64_content}' | base64 -d > '{sandbox_path}'"
                )

                result = await self.container_manager.execute_in_sandbox(
                    container_id,
                    write_cmd,
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
