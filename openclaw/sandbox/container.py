"""
Container Manager - Docker container lifecycle management.
Creates ephemeral, isolated containers for safe code execution.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Optional

from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.sandbox.container")


class ContainerManager:
    """
    Manages Docker containers for sandboxed execution.

    Features:
    - Ephemeral containers (destroyed after use)
    - Resource limits (CPU, memory, time)
    - Network isolation
    - Volume mounting for workspace
    - Automatic cleanup
    """

    def __init__(self):
        self.settings = get_settings()
        self._docker_client = None
        self._active_containers: dict[str, str] = {}  # session_id -> container_id
        self._initialized = False

    async def initialize(self):
        """Initialize Docker client."""
        if self._initialized:
            return

        try:
            import docker
            self._docker_client = docker.from_env()
            self._docker_client.ping()
            self._initialized = True
            logger.info("Container manager initialized")
        except ImportError:
            logger.error("docker package not installed. Run: pip install docker")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Docker: {e}")
            raise

    async def create_sandbox(self, session_id: str = None) -> str:
        """
        Create a new sandbox container.

        Args:
            session_id: Optional session ID to associate with container

        Returns:
            Container ID
        """
        await self.initialize()

        session_id = session_id or str(uuid.uuid4())[:8]
        container_name = f"openclaw-sandbox-{session_id}"

        # Get sandbox configuration
        image = self.settings.get("sandbox.image", "python:3.11-slim")
        memory_limit = self.settings.get("sandbox.memory_limit", "256m")
        cpu_quota = self.settings.get("sandbox.cpu_quota", 50000)  # 50% of one CPU
        timeout = self.settings.get("sandbox.timeout", 30)

        # Create workspace directory
        workspace_base = Path(self.settings.get("sandbox.workspace", "/tmp/openclaw-sandbox"))
        workspace_base.mkdir(parents=True, exist_ok=True)
        workspace = workspace_base / session_id
        workspace.mkdir(exist_ok=True)

        try:
            container = self._docker_client.containers.create(
                image=image,
                name=container_name,
                command="sleep infinity",  # Keep alive until killed
                detach=True,
                mem_limit=memory_limit,
                cpu_quota=cpu_quota,
                network_mode="none",  # No network access by default
                security_opt=["no-new-privileges:true"],
                cap_drop=["ALL"],  # Drop all capabilities
                read_only=False,  # Allow writes to mounted workspace
                volumes={
                    str(workspace): {"bind": "/workspace", "mode": "rw"}
                },
                working_dir="/workspace",
                environment={
                    "SANDBOX": "1",
                    "SESSION_ID": session_id,
                },
                labels={
                    "openclaw.sandbox": "true",
                    "openclaw.session": session_id,
                }
            )

            container.start()
            self._active_containers[session_id] = container.id

            logger.info(f"Created sandbox container: {container_name}")
            return container.id

        except Exception as e:
            logger.error(f"Failed to create sandbox container: {e}")
            raise

    async def execute_in_sandbox(
        self,
        container_id: str,
        command: str,
        timeout: int = None,
        user: str = "nobody",
    ) -> dict:
        """
        Execute a command in a sandbox container.

        Args:
            container_id: Container ID
            command: Command to execute
            timeout: Execution timeout in seconds
            user: User to run as (default: nobody)

        Returns:
            Execution result with stdout, stderr, exit_code
        """
        await self.initialize()

        timeout = timeout or self.settings.get("sandbox.timeout", 30)

        try:
            container = self._docker_client.containers.get(container_id)

            # Execute command
            exec_result = container.exec_run(
                cmd=["sh", "-c", command],
                user=user,
                workdir="/workspace",
                demux=True,  # Separate stdout/stderr
                environment={"PATH": "/usr/local/bin:/usr/bin:/bin"},
            )

            stdout = exec_result.output[0] if exec_result.output[0] else b""
            stderr = exec_result.output[1] if exec_result.output[1] else b""

            return {
                "success": exec_result.exit_code == 0,
                "exit_code": exec_result.exit_code,
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
            }

        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return {
                "success": False,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
            }

    async def execute_python(
        self,
        container_id: str,
        code: str,
        timeout: int = None,
    ) -> dict:
        """
        Execute Python code in a sandbox container.

        Args:
            container_id: Container ID
            code: Python code to execute
            timeout: Execution timeout

        Returns:
            Execution result
        """
        # Write code to a temporary file in the container
        escaped_code = code.replace("'", "'\\''")
        write_cmd = f"echo '{escaped_code}' > /workspace/_script.py"

        await self.execute_in_sandbox(container_id, write_cmd)

        # Execute the Python script
        return await self.execute_in_sandbox(
            container_id,
            "python3 /workspace/_script.py",
            timeout=timeout,
        )

    async def copy_to_sandbox(self, container_id: str, local_path: str, container_path: str):
        """Copy a file into the sandbox container."""
        await self.initialize()

        import tarfile
        import io

        container = self._docker_client.containers.get(container_id)

        # Create tar archive of the file
        file_data = Path(local_path).read_bytes()
        tarstream = io.BytesIO()
        tar = tarfile.open(fileobj=tarstream, mode='w')
        tarinfo = tarfile.TarInfo(name=Path(container_path).name)
        tarinfo.size = len(file_data)
        tar.addfile(tarinfo, io.BytesIO(file_data))
        tar.close()
        tarstream.seek(0)

        container.put_archive(str(Path(container_path).parent), tarstream)

    async def copy_from_sandbox(self, container_id: str, container_path: str) -> bytes:
        """Copy a file from the sandbox container."""
        await self.initialize()

        import tarfile
        import io

        container = self._docker_client.containers.get(container_id)

        bits, _ = container.get_archive(container_path)
        tarstream = io.BytesIO()
        for chunk in bits:
            tarstream.write(chunk)
        tarstream.seek(0)

        tar = tarfile.open(fileobj=tarstream, mode='r')
        member = tar.getmembers()[0]
        f = tar.extractfile(member)
        return f.read() if f else b""

    async def destroy_sandbox(self, container_id: str):
        """Destroy a sandbox container."""
        await self.initialize()

        try:
            container = self._docker_client.containers.get(container_id)
            container.stop(timeout=5)
            container.remove(force=True)

            # Remove from active containers
            for session_id, cid in list(self._active_containers.items()):
                if cid == container_id:
                    del self._active_containers[session_id]
                    break

            logger.info(f"Destroyed sandbox container: {container_id}")

        except Exception as e:
            logger.error(f"Failed to destroy container {container_id}: {e}")

    async def cleanup_all(self):
        """Cleanup all sandbox containers."""
        await self.initialize()

        # Find all openclaw sandbox containers
        containers = self._docker_client.containers.list(
            filters={"label": "openclaw.sandbox=true"},
            all=True,
        )

        for container in containers:
            try:
                container.stop(timeout=5)
                container.remove(force=True)
            except Exception as e:
                logger.warning(f"Failed to cleanup container {container.id}: {e}")

        self._active_containers.clear()
        logger.info(f"Cleaned up {len(containers)} sandbox containers")

    async def get_sandbox_for_session(self, session_id: str) -> Optional[str]:
        """Get or create a sandbox for a session."""
        if session_id in self._active_containers:
            # Verify container still exists
            try:
                self._docker_client.containers.get(self._active_containers[session_id])
                return self._active_containers[session_id]
            except Exception:
                del self._active_containers[session_id]

        return await self.create_sandbox(session_id)

    @property
    def active_count(self) -> int:
        return len(self._active_containers)
