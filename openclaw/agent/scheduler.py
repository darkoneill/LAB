"""
Chronotaches - Agentic Cron Scheduler.

Allows the agent to self-schedule future tasks via schedule_task().
A background asyncio loop picks up scheduled tasks and fires them
at the appropriate time, notifying the user via WebSocket.

Usage:
    scheduler = TaskScheduler(brain, ws_manager)
    task_id = scheduler.schedule(description="Run tests", delay_minutes=10)
    await scheduler.start()  # starts the background loop
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from openclaw.agent.brain import AgentBrain

logger = logging.getLogger("openclaw.agent.scheduler")


@dataclass
class ScheduledTask:
    """A task scheduled for future execution."""
    task_id: str = field(default_factory=lambda: f"cron_{uuid.uuid4().hex[:8]}")
    description: str = ""
    session_id: str = ""
    fire_at: float = 0.0  # Unix timestamp
    status: str = "pending"  # pending, running, completed, failed, cancelled
    result: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "session_id": self.session_id,
            "fire_at": self.fire_at,
            "status": self.status,
            "result": self.result[:500] if self.result else "",
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "delay_remaining_s": max(0, round(self.fire_at - time.time(), 1)),
        }


class TaskScheduler:
    """
    Background task scheduler for autonomous agent operations.

    The agent can call schedule_task() to queue a task for future execution.
    A background loop checks every 5 seconds for tasks that are due.
    """

    def __init__(self, brain: "AgentBrain" = None, ws_manager=None):
        self.brain = brain
        self._ws_manager = ws_manager
        self._tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        # Guard: max simultaneous scheduled tasks
        self._max_pending = 20

    def schedule(
        self,
        description: str,
        delay_minutes: float,
        session_id: str = "",
    ) -> str:
        """
        Schedule a task for future execution.

        Args:
            description: What the agent should do when the task fires.
            delay_minutes: Minutes from now until execution.
            session_id: Session context for the task.

        Returns:
            task_id of the scheduled task.
        """
        pending_count = sum(
            1 for t in self._tasks.values() if t.status == "pending"
        )
        if pending_count >= self._max_pending:
            raise ValueError(
                f"Too many pending tasks ({pending_count}). "
                f"Max is {self._max_pending}."
            )

        if delay_minutes < 0.1:
            delay_minutes = 0.1  # Minimum 6 seconds

        task = ScheduledTask(
            description=description,
            session_id=session_id,
            fire_at=time.time() + (delay_minutes * 60),
        )
        self._tasks[task.task_id] = task
        logger.info(
            f"Task scheduled: {task.task_id} in {delay_minutes}min "
            f"- '{description[:60]}'"
        )
        return task.task_id

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending scheduled task."""
        task = self._tasks.get(task_id)
        if task and task.status == "pending":
            task.status = "cancelled"
            logger.info(f"Task cancelled: {task_id}")
            return True
        return False

    def list_tasks(self, include_done: bool = False) -> list[dict]:
        """List scheduled tasks."""
        tasks = self._tasks.values()
        if not include_done:
            tasks = [t for t in tasks if t.status in ("pending", "running")]
        return [t.to_dict() for t in sorted(tasks, key=lambda t: t.fire_at)]

    async def start(self):
        """Start the background scheduler loop."""
        if self._running:
            return
        self._running = True
        self._loop_task = asyncio.create_task(self._scheduler_loop())
        logger.info("TaskScheduler started")

    async def stop(self):
        """Stop the background scheduler loop."""
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("TaskScheduler stopped")

    async def _scheduler_loop(self):
        """Background loop: check for due tasks every 5 seconds."""
        while self._running:
            try:
                now = time.time()
                for task in list(self._tasks.values()):
                    if task.status == "pending" and task.fire_at <= now:
                        await self._execute_task(task)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")

            await asyncio.sleep(5)

    async def _execute_task(self, task: ScheduledTask):
        """Execute a due scheduled task."""
        task.status = "running"
        logger.info(f"Executing scheduled task: {task.task_id}")

        # Notify UI
        await self._notify("scheduled_task_started", task)

        try:
            if not self.brain:
                raise RuntimeError("No agent brain configured")

            result = await self.brain.generate(
                messages=[{
                    "role": "user",
                    "content": (
                        f"[TACHE PLANIFIEE - Chronotache]\n"
                        f"Cette tache a ete programmee par toi-meme il y a "
                        f"{round((time.time() - task.created_at) / 60, 1)} minutes.\n\n"
                        f"Description: {task.description}\n\n"
                        f"Execute cette tache maintenant."
                    ),
                }],
                max_tokens=4096,
            )
            task.result = result.get("content", "")
            task.status = "completed"
            task.completed_at = time.time()
            logger.info(f"Scheduled task completed: {task.task_id}")

        except Exception as e:
            task.status = "failed"
            task.result = f"Error: {e}"
            task.completed_at = time.time()
            logger.error(f"Scheduled task failed: {task.task_id}: {e}")

        # Notify UI with result
        await self._notify("scheduled_task_completed", task)

    async def _notify(self, event_type: str, task: ScheduledTask):
        """Send a WebSocket notification about a scheduled task event."""
        if not self._ws_manager:
            return
        try:
            await self._ws_manager.broadcast({
                "type": event_type,
                "task": task.to_dict(),
            })
        except Exception:
            pass
