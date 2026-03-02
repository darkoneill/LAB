"""
Tests for openclaw/agent/scheduler.py — TaskScheduler.
All I/O and time functions are mocked; no real delays.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw.agent.scheduler import ScheduledTask, TaskScheduler


# ── ScheduledTask dataclass ─────────────────────────────────


class TestScheduledTask:
    def test_default_fields(self):
        t = ScheduledTask()
        assert t.task_id.startswith("cron_")
        assert t.status == "pending"
        assert t.description == ""
        assert t.result == ""

    def test_to_dict_contains_all_keys(self):
        t = ScheduledTask(description="run tests", fire_at=time.time() + 60)
        d = t.to_dict()
        assert "task_id" in d
        assert d["description"] == "run tests"
        assert d["status"] == "pending"
        assert "delay_remaining_s" in d
        assert d["delay_remaining_s"] >= 0

    def test_to_dict_truncates_long_result(self):
        t = ScheduledTask(result="x" * 1000)
        d = t.to_dict()
        assert len(d["result"]) == 500

    def test_to_dict_empty_result(self):
        t = ScheduledTask(result="")
        d = t.to_dict()
        assert d["result"] == ""


# ── schedule / cancel ────────────────────────────────────────


class TestScheduleTask:
    def test_schedule_returns_task_id(self):
        sched = TaskScheduler()
        tid = sched.schedule("do stuff", delay_minutes=5)
        assert tid.startswith("cron_")

    def test_schedule_stores_task(self):
        sched = TaskScheduler()
        tid = sched.schedule("build project", delay_minutes=1)
        assert tid in sched._tasks
        assert sched._tasks[tid].description == "build project"
        assert sched._tasks[tid].status == "pending"

    def test_schedule_minimum_delay(self):
        sched = TaskScheduler()
        tid = sched.schedule("fast task", delay_minutes=0.01)
        task = sched._tasks[tid]
        # Minimum enforced to 0.1 minutes (6 seconds), allow small float drift
        assert task.fire_at >= task.created_at + 5.9

    def test_schedule_respects_max_pending(self):
        sched = TaskScheduler()
        sched._max_pending = 3
        sched.schedule("t1", delay_minutes=5)
        sched.schedule("t2", delay_minutes=5)
        sched.schedule("t3", delay_minutes=5)
        with pytest.raises(ValueError, match="Too many pending tasks"):
            sched.schedule("t4", delay_minutes=5)

    def test_cancel_pending_task(self):
        sched = TaskScheduler()
        tid = sched.schedule("cancel me", delay_minutes=5)
        assert sched.cancel(tid) is True
        assert sched._tasks[tid].status == "cancelled"

    def test_cancel_nonexistent_returns_false(self):
        sched = TaskScheduler()
        assert sched.cancel("cron_nope") is False

    def test_cancel_already_running_returns_false(self):
        sched = TaskScheduler()
        tid = sched.schedule("running", delay_minutes=5)
        sched._tasks[tid].status = "running"
        assert sched.cancel(tid) is False

    def test_cancel_allows_new_schedule(self):
        sched = TaskScheduler()
        sched._max_pending = 1
        tid = sched.schedule("t1", delay_minutes=5)
        sched.cancel(tid)
        # Now should be able to schedule another
        tid2 = sched.schedule("t2", delay_minutes=5)
        assert tid2 != tid


# ── list_tasks ───────────────────────────────────────────────


class TestListTasks:
    def test_list_active_tasks(self):
        sched = TaskScheduler()
        sched.schedule("a", delay_minutes=10)
        sched.schedule("b", delay_minutes=5)
        tasks = sched.list_tasks()
        assert len(tasks) == 2
        # Sorted by fire_at (b first since shorter delay)
        assert tasks[0]["description"] == "b"
        assert tasks[1]["description"] == "a"

    def test_list_excludes_done_by_default(self):
        sched = TaskScheduler()
        tid = sched.schedule("done", delay_minutes=1)
        sched._tasks[tid].status = "completed"
        assert len(sched.list_tasks()) == 0

    def test_list_includes_done_when_requested(self):
        sched = TaskScheduler()
        tid = sched.schedule("done", delay_minutes=1)
        sched._tasks[tid].status = "completed"
        assert len(sched.list_tasks(include_done=True)) == 1


# ── start / stop lifecycle ──────────────────────────────────


class TestSchedulerLifecycle:
    @pytest.mark.asyncio
    async def test_start_sets_running(self):
        sched = TaskScheduler()
        with patch.object(sched, "_scheduler_loop", new_callable=AsyncMock):
            await sched.start()
            assert sched._running is True
            assert sched._loop_task is not None
            await sched.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_loop(self):
        sched = TaskScheduler()
        with patch.object(sched, "_scheduler_loop", new_callable=AsyncMock):
            await sched.start()
            await sched.stop()
            assert sched._running is False

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self):
        sched = TaskScheduler()
        with patch.object(sched, "_scheduler_loop", new_callable=AsyncMock):
            await sched.start()
            first_task = sched._loop_task
            await sched.start()  # Should not create a new task
            assert sched._loop_task is first_task
            await sched.stop()


# ── _execute_task ────────────────────────────────────────────


class TestExecuteTask:
    @pytest.mark.asyncio
    async def test_execute_calls_brain(self):
        brain = AsyncMock()
        brain.generate.return_value = {"content": "Done!"}
        sched = TaskScheduler(brain=brain)
        task = ScheduledTask(description="do something", fire_at=0)

        await sched._execute_task(task)

        assert task.status == "completed"
        assert task.result == "Done!"
        assert task.completed_at is not None
        brain.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_without_brain_fails(self):
        sched = TaskScheduler(brain=None)
        task = ScheduledTask(description="fail", fire_at=0)

        await sched._execute_task(task)

        assert task.status == "failed"
        assert "No agent brain" in task.result

    @pytest.mark.asyncio
    async def test_execute_brain_error_marks_failed(self):
        brain = AsyncMock()
        brain.generate.side_effect = RuntimeError("LLM timeout")
        sched = TaskScheduler(brain=brain)
        task = ScheduledTask(description="boom", fire_at=0)

        await sched._execute_task(task)

        assert task.status == "failed"
        assert "LLM timeout" in task.result

    @pytest.mark.asyncio
    async def test_execute_notifies_ws(self):
        brain = AsyncMock()
        brain.generate.return_value = {"content": "ok"}
        ws = AsyncMock()
        sched = TaskScheduler(brain=brain, ws_manager=ws)
        task = ScheduledTask(description="notify me", fire_at=0)

        await sched._execute_task(task)

        assert ws.broadcast.call_count == 2  # started + completed
        first_call = ws.broadcast.call_args_list[0][0][0]
        assert first_call["type"] == "scheduled_task_started"
        second_call = ws.broadcast.call_args_list[1][0][0]
        assert second_call["type"] == "scheduled_task_completed"


# ── _scheduler_loop (integration-ish) ───────────────────────


class TestSchedulerLoop:
    @pytest.mark.asyncio
    async def test_loop_picks_up_due_task(self):
        brain = AsyncMock()
        brain.generate.return_value = {"content": "executed"}
        sched = TaskScheduler(brain=brain)
        tid = sched.schedule("due now", delay_minutes=0.1)
        # Make it already due
        sched._tasks[tid].fire_at = time.time() - 1

        # Patch sleep to stop loop after one iteration
        call_count = 0
        async def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            sched._running = False

        with patch("openclaw.agent.scheduler.asyncio.sleep", side_effect=fake_sleep):
            sched._running = True
            await sched._scheduler_loop()

        assert sched._tasks[tid].status == "completed"

    @pytest.mark.asyncio
    async def test_loop_error_does_not_crash(self):
        """A failing task should not crash the scheduler loop."""
        brain = AsyncMock()
        brain.generate.side_effect = Exception("kaboom")
        sched = TaskScheduler(brain=brain)
        tid = sched.schedule("fail task", delay_minutes=0.1)
        sched._tasks[tid].fire_at = time.time() - 1

        async def fake_sleep(seconds):
            sched._running = False

        with patch("openclaw.agent.scheduler.asyncio.sleep", side_effect=fake_sleep):
            sched._running = True
            await sched._scheduler_loop()

        # Task failed but scheduler didn't crash
        assert sched._tasks[tid].status == "failed"

    @pytest.mark.asyncio
    async def test_loop_skips_non_pending_tasks(self):
        brain = AsyncMock()
        brain.generate.return_value = {"content": "ok"}
        sched = TaskScheduler(brain=brain)
        tid = sched.schedule("skip me", delay_minutes=0.1)
        sched._tasks[tid].fire_at = time.time() - 1
        sched._tasks[tid].status = "cancelled"

        async def fake_sleep(seconds):
            sched._running = False

        with patch("openclaw.agent.scheduler.asyncio.sleep", side_effect=fake_sleep):
            sched._running = True
            await sched._scheduler_loop()

        # Should still be cancelled, not executed
        assert sched._tasks[tid].status == "cancelled"
        brain.generate.assert_not_called()
