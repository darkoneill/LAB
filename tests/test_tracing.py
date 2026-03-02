"""
Tests for openclaw/tracing/recorder.py — Span, Trace, TraceRecorder.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from openclaw.tracing.recorder import Span, SpanKind, Trace, TraceRecorder


# ── Span ────────────────────────────────────────────────────


class TestSpan:
    def test_default_fields(self):
        s = Span()
        assert len(s.span_id) == 12
        assert s.status == "ok"
        assert s.end_time is None
        assert s.duration_ms is None

    def test_finish_sets_duration(self):
        s = Span(start_time=time.time() - 0.1)
        s.finish()
        assert s.end_time is not None
        assert s.duration_ms >= 0
        assert s.status == "ok"

    def test_finish_with_error_status(self):
        s = Span()
        s.finish(status="error")
        assert s.status == "error"

    def test_add_event(self):
        s = Span()
        s.add_event("token_count", {"input": 100, "output": 50})
        assert len(s.events) == 1
        assert s.events[0]["name"] == "token_count"
        assert s.events[0]["attributes"]["input"] == 100

    def test_to_dict(self):
        s = Span(kind=SpanKind.LLM_CALL, name="generate")
        d = s.to_dict()
        assert d["kind"] == "llm_call"
        assert d["name"] == "generate"
        assert "span_id" in d

    def test_captures_latency(self):
        s = Span(start_time=1000.0)
        with patch("openclaw.tracing.recorder.time") as mock_time:
            mock_time.time.return_value = 1000.150
            s.finish()
        assert s.duration_ms == 150.0


# ── Trace ───────────────────────────────────────────────────


class TestTrace:
    def test_default_fields(self):
        t = Trace()
        assert t.trace_id.startswith("trace_")
        assert t.status == "in_progress"
        assert t.spans == []

    def test_add_span_sets_trace_id(self):
        t = Trace()
        s = Span(kind=SpanKind.REQUEST, name="incoming")
        t.add_span(s)
        assert s.trace_id == t.trace_id
        assert len(t.spans) == 1

    def test_finish_sets_completed(self):
        t = Trace(start_time=time.time() - 0.05)
        t.finish()
        assert t.status == "completed"
        assert t.duration_ms >= 0

    def test_to_dict(self):
        t = Trace(user_input="hello", session_id="s1")
        s = Span(kind=SpanKind.LLM_CALL, name="gen")
        t.add_span(s)
        d = t.to_dict()
        assert d["user_input"] == "hello"
        assert len(d["spans"]) == 1

    def test_to_dict_truncates_long_fields(self):
        t = Trace(user_input="x" * 1000, final_response="y" * 1000)
        d = t.to_dict()
        assert len(d["user_input"]) == 500
        assert len(d["final_response"]) == 500

    def test_summary(self):
        t = Trace(user_input="What is Python?", session_id="s1")
        t.add_span(Span())
        t.add_span(Span())
        s = t.summary()
        assert s["span_count"] == 2
        assert s["user_input"] == "What is Python?"

    def test_nested_spans(self):
        t = Trace()
        parent = Span(kind=SpanKind.REQUEST, name="main")
        t.add_span(parent)
        child = Span(kind=SpanKind.LLM_CALL, name="gen", parent_span_id=parent.span_id)
        t.add_span(child)
        assert child.parent_span_id == parent.span_id
        assert len(t.spans) == 2


# ── TraceRecorder ───────────────────────────────────────────


def _make_recorder(enabled=True, persist=False, max_traces=100):
    settings = MagicMock()
    settings.get = lambda k, d=None: {
        "tracing.enabled": enabled,
        "tracing.max_traces": max_traces,
        "tracing.persist": persist,
        "tracing.store_path": "/tmp/test_traces",
    }.get(k, d)

    with patch("openclaw.tracing.recorder.get_settings", return_value=settings):
        return TraceRecorder()


class TestTraceRecorder:
    def test_start_trace(self):
        rec = _make_recorder()
        t = rec.start_trace("hello", session_id="s1")
        assert t.user_input == "hello"
        assert t.trace_id in rec._active_traces

    def test_start_trace_disabled(self):
        rec = _make_recorder(enabled=False)
        t = rec.start_trace("hello")
        # Returns a trace but doesn't store it
        assert t.user_input == "hello"
        assert t.trace_id not in rec._active_traces

    def test_start_span(self):
        rec = _make_recorder()
        t = rec.start_trace("test")
        s = rec.start_span(t, SpanKind.LLM_CALL, "generate")
        assert s.kind == SpanKind.LLM_CALL
        assert s.trace_id == t.trace_id
        assert len(t.spans) == 1

    def test_finish_span(self):
        rec = _make_recorder()
        t = rec.start_trace("test")
        s = rec.start_span(t, SpanKind.LLM_CALL, "gen")
        rec.finish_span(s, status="ok", attributes={"tokens": 500})
        assert s.status == "ok"
        assert s.attributes["tokens"] == 500
        assert s.duration_ms is not None

    def test_finish_trace(self):
        rec = _make_recorder()
        t = rec.start_trace("test")
        rec.finish_trace(t, final_response="Done")
        assert t.status == "completed"
        assert t.final_response == "Done"
        assert t.trace_id not in rec._active_traces
        assert len(rec._traces) == 1

    def test_get_trace_from_buffer(self):
        rec = _make_recorder()
        t = rec.start_trace("lookup")
        rec.finish_trace(t, final_response="found")
        result = rec.get_trace(t.trace_id)
        assert result is not None
        assert result["user_input"] == "lookup"

    def test_get_trace_not_found(self):
        rec = _make_recorder()
        assert rec.get_trace("nonexistent") is None

    def test_list_traces(self):
        rec = _make_recorder()
        for i in range(5):
            t = rec.start_trace(f"q{i}", session_id="s1")
            rec.finish_trace(t)
        summaries = rec.list_traces()
        assert len(summaries) == 5

    def test_list_traces_filter_session(self):
        rec = _make_recorder()
        t1 = rec.start_trace("a", session_id="s1")
        rec.finish_trace(t1)
        t2 = rec.start_trace("b", session_id="s2")
        rec.finish_trace(t2)
        assert len(rec.list_traces(session_id="s1")) == 1

    def test_list_traces_filter_status(self):
        rec = _make_recorder()
        t1 = rec.start_trace("ok")
        rec.finish_trace(t1, status="completed")
        t2 = rec.start_trace("bad")
        rec.finish_trace(t2, status="error")
        assert len(rec.list_traces(status="error")) == 1

    def test_ring_buffer_eviction(self):
        rec = _make_recorder(max_traces=3)
        traces = []
        for i in range(5):
            t = rec.start_trace(f"q{i}")
            rec.finish_trace(t)
            traces.append(t)
        # Only 3 most recent should remain
        assert len(rec._traces) == 3
        # Oldest two should be evicted from index
        assert traces[0].trace_id not in rec._trace_index
        assert traces[1].trace_id not in rec._trace_index
        assert traces[4].trace_id in rec._trace_index

    def test_search_traces(self):
        rec = _make_recorder()
        t1 = rec.start_trace("How does Python work?")
        rec.finish_trace(t1)
        t2 = rec.start_trace("What is Rust?")
        rec.finish_trace(t2)
        results = rec.search_traces("python")
        assert len(results) == 1
        assert results[0]["user_input"].startswith("How does Python")

    def test_search_traces_case_insensitive(self):
        rec = _make_recorder()
        t = rec.start_trace("JAVASCRIPT frameworks")
        rec.finish_trace(t)
        assert len(rec.search_traces("javascript")) == 1


class TestTraceRecorderStats:
    def test_get_stats_empty(self):
        rec = _make_recorder()
        stats = rec.get_stats()
        assert stats["total_traces"] == 0
        assert stats["avg_duration_ms"] == 0

    def test_get_stats_with_traces(self):
        rec = _make_recorder()
        for i in range(3):
            t = rec.start_trace(f"q{i}")
            rec.finish_trace(t, status="completed")
        t_err = rec.start_trace("err")
        rec.finish_trace(t_err, status="error")

        stats = rec.get_stats()
        assert stats["total_traces"] == 4
        assert stats["completed"] == 3
        assert stats["errors"] == 1


class TestTraceRecorderPersistence:
    def test_persist_trace_to_file(self, tmp_path):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "tracing.enabled": True,
            "tracing.max_traces": 100,
            "tracing.persist": True,
            "tracing.store_path": str(tmp_path / "traces"),
        }.get(k, d)

        with patch("openclaw.tracing.recorder.get_settings", return_value=settings):
            rec = TraceRecorder()

        t = rec.start_trace("persist me")
        rec.start_span(t, SpanKind.LLM_CALL, "gen")
        rec.finish_trace(t, final_response="persisted")

        filepath = tmp_path / "traces" / f"{t.trace_id}.json"
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data["user_input"] == "persist me"
        assert len(data["spans"]) == 1

    def test_get_trace_from_disk(self, tmp_path):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "tracing.enabled": True,
            "tracing.max_traces": 1,
            "tracing.persist": True,
            "tracing.store_path": str(tmp_path / "traces"),
        }.get(k, d)

        with patch("openclaw.tracing.recorder.get_settings", return_value=settings):
            rec = TraceRecorder()

        # Create and evict a trace
        t1 = rec.start_trace("first")
        rec.finish_trace(t1)
        t2 = rec.start_trace("second")
        rec.finish_trace(t2)

        # t1 evicted from memory but still on disk
        result = rec.get_trace(t1.trace_id)
        assert result is not None
        assert result["user_input"] == "first"


class TestTraceRecorderReplay:
    def test_replay_context_basic(self):
        rec = _make_recorder()
        t = rec.start_trace("fix the bug")
        s1 = rec.start_span(t, SpanKind.LLM_CALL, "generate")
        rec.finish_span(s1, attributes={"response": "Here is the fix"})
        s2 = rec.start_span(t, SpanKind.TOOL_EXEC, "run_tests")
        rec.finish_span(s2, status="error", attributes={"error": "test failed"})
        rec.finish_trace(t, status="error")

        ctx = rec.get_replay_context(t.trace_id)
        assert ctx is not None
        assert ctx["user_input"] == "fix the bug"
        assert ctx["failed_span"] is not None
        assert ctx["failed_span"]["name"] == "run_tests"
        assert len(ctx["replay_messages"]) >= 2

    def test_replay_context_not_found(self):
        rec = _make_recorder()
        assert rec.get_replay_context("nonexistent") is None


class TestSanitizeTraceId:
    def test_normal_id(self):
        assert TraceRecorder._sanitize_trace_id("trace_abc123") == "trace_abc123"

    def test_path_traversal(self):
        assert ".." not in TraceRecorder._sanitize_trace_id("../../etc/passwd")

    def test_special_chars_stripped(self):
        assert TraceRecorder._sanitize_trace_id("a/b\\c<d>e") == "abcde"


class TestSpanKind:
    def test_all_span_kinds_are_strings(self):
        for kind in SpanKind:
            assert isinstance(kind.value, str)

    def test_known_kinds(self):
        assert SpanKind.REQUEST == "request"
        assert SpanKind.LLM_CALL == "llm_call"
        assert SpanKind.TOOL_EXEC == "tool_exec"
        assert SpanKind.SELF_HEAL == "self_heal"


class TestRecorderThreadSafety:
    def test_concurrent_traces(self):
        """Multiple threads creating traces should not corrupt state."""
        rec = _make_recorder(max_traces=500)

        def create_trace(i):
            t = rec.start_trace(f"query_{i}")
            rec.start_span(t, SpanKind.LLM_CALL, f"gen_{i}")
            rec.finish_trace(t)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(create_trace, range(50)))

        # All 50 traces should be present
        assert len(rec._traces) == 50
