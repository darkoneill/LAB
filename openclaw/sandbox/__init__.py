"""
Sandbox - Isolated execution environment for dangerous operations.
Uses Docker containers for secure code execution.
"""

from openclaw.sandbox.executor import SandboxExecutor
from openclaw.sandbox.container import ContainerManager

__all__ = ["SandboxExecutor", "ContainerManager"]
