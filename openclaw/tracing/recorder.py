"""
Trace Recorder - Black Box for the agent pipeline.

Records structured traces of every step:
  Input -> Retrieval (similarity scores) -> Prompt sent to LLM -> Code generated
  -> Sandbox output -> Final response

Traces are stored as structured JSON and can be replayed in the UI.
Compatible with OpenTelemetry span conventions for future integration.
"""

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.tracing.recorder")


class SpanKind(str, Enum):
    """Type of trace span, aligned with OpenTelemetry conventions."""
    REQUEST = "request"           # Incoming user request
    RETRIEVAL = "retrieval"       # Memory/RAG retrieval step
    LLM_CALL = "llm_call"        # LLM generation call
    TOOL_EXEC = "tool_exec"       # Tool/sandbox execution
    SELF_HEAL = "self_heal"       # Self-healing correction attempt
    DELEGATION = "delegation"     # Agent delegation/swarm
    MCP_CALL = "mcp_call"         # MCP tool invocation
    APPROVAL = "approval"         # Human-in-the-loop approval
    RESPONSE = "response"         # Final response to user


@dataclass
class Span:
    """A single span in a trace, representing one pipeline step."""
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    trace_id: str = ""
    parent_span_id: Optional[str] = None
    kind: str = SpanKind.REQUEST
    name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "ok"  # ok, error, timeout
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    def finish(self, status: str = "ok"):
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.status = status

    def add_event(self, name: str, attributes: dict = None):
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trace:
    """A complete trace representing one user request through the pipeline."""
    trace_id: str = field(default_factory=lambda: f"trace_{uuid.uuid4().hex[:16]}")
    session_id: str = ""
    user_input: str = ""
    final_response: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "in_progress"  # in_progress, completed, error
    spans: list[Span] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def finish(self, status: str = "completed"):
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.status = status

    def add_span(self, span: Span) -> Span:
        span.trace_id = self.trace_id
        self.spans.append(span)
        return span

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_input": self.user_input[:500],
            "final_response": self.final_response[:500],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
        }

    def summary(self) -> dict:
        """Compact summary for listing."""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_input": self.user_input[:100],
            "status": self.status,
            "duration_ms": self.duration_ms,
            "span_count": len(self.spans),
            "start_time": self.start_time,
        }


class TraceRecorder:
    """
    Central trace recorder that stores and manages pipeline traces.

    Features:
    - In-memory ring buffer for recent traces (fast access)
    - Optional JSON file persistence for long-term storage
    - Trace search and replay
    - Statistics and performance metrics
    """

    def __init__(self):
        self.settings = get_settings()
        self._enabled = self.settings.get("tracing.enabled", True)
        self._max_traces = self.settings.get("tracing.max_traces", 500)
        self._persist = self.settings.get("tracing.persist", True)
        self._traces: deque[Trace] = deque(maxlen=self._max_traces)
        self._active_traces: dict[str, Trace] = {}
        self._trace_dir = Path(
            self.settings.get("tracing.store_path", "logs/traces")
        )
        if self._persist:
            self._trace_dir.mkdir(parents=True, exist_ok=True)

    def start_trace(
        self,
        user_input: str,
        session_id: str = "",
        metadata: dict = None,
    ) -> Trace:
        """Start a new trace for a user request."""
        if not self._enabled:
            return Trace(user_input=user_input, session_id=session_id)

        trace = Trace(
            user_input=user_input,
            session_id=session_id,
            metadata=metadata or {},
        )
        self._active_traces[trace.trace_id] = trace
        logger.debug(f"Trace started: {trace.trace_id}")
        return trace

    def start_span(
        self,
        trace: Trace,
        kind: SpanKind,
        name: str,
        parent_span_id: str = None,
        attributes: dict = None,
    ) -> Span:
        """Add a new span to an active trace."""
        span = Span(
            kind=kind,
            name=name,
            parent_span_id=parent_span_id,
            attributes=attributes or {},
        )
        trace.add_span(span)
        return span

    def finish_span(self, span: Span, status: str = "ok", attributes: dict = None):
        """Complete a span with final status and optional extra attributes."""
        if attributes:
            span.attributes.update(attributes)
        span.finish(status)

    def finish_trace(
        self,
        trace: Trace,
        final_response: str = "",
        status: str = "completed",
    ):
        """Complete a trace and persist it."""
        trace.final_response = final_response
        trace.finish(status)

        self._active_traces.pop(trace.trace_id, None)
        self._traces.append(trace)

        if self._persist:
            self._persist_trace(trace)

        logger.debug(
            f"Trace completed: {trace.trace_id} "
            f"({trace.duration_ms}ms, {len(trace.spans)} spans)"
        )

    def _persist_trace(self, trace: Trace):
        """Write trace to JSON file for long-term storage."""
        try:
            filepath = self._trace_dir / f"{trace.trace_id}.json"
            filepath.write_text(
                json.dumps(trace.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error(f"Failed to persist trace {trace.trace_id}: {e}")

    def get_trace(self, trace_id: str) -> Optional[dict]:
        """Retrieve a trace by ID (memory first, then disk)."""
        # Check active
        if trace_id in self._active_traces:
            return self._active_traces[trace_id].to_dict()

        # Check ring buffer
        for t in self._traces:
            if t.trace_id == trace_id:
                return t.to_dict()

        # Check persisted
        filepath = self._trace_dir / f"{trace_id}.json"
        if filepath.exists():
            try:
                return json.loads(filepath.read_text(encoding="utf-8"))
            except Exception:
                pass

        return None

    def list_traces(
        self,
        session_id: str = None,
        status: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List trace summaries with optional filters."""
        traces = list(self._traces)

        if session_id:
            traces = [t for t in traces if t.session_id == session_id]
        if status:
            traces = [t for t in traces if t.status == status]

        # Most recent first
        traces.sort(key=lambda t: t.start_time, reverse=True)
        page = traces[offset:offset + limit]
        return [t.summary() for t in page]

    def get_stats(self) -> dict:
        """Get tracing statistics."""
        completed = [t for t in self._traces if t.status == "completed"]
        errors = [t for t in self._traces if t.status == "error"]
        durations = [t.duration_ms for t in completed if t.duration_ms]

        return {
            "total_traces": len(self._traces),
            "active_traces": len(self._active_traces),
            "completed": len(completed),
            "errors": len(errors),
            "avg_duration_ms": round(sum(durations) / len(durations), 2) if durations else 0,
            "p95_duration_ms": round(
                sorted(durations)[int(len(durations) * 0.95)] if durations else 0, 2
            ),
            "max_traces_capacity": self._max_traces,
        }

    def search_traces(self, query: str, limit: int = 20) -> list[dict]:
        """Search traces by user input content."""
        query_lower = query.lower()
        results = []
        for trace in reversed(self._traces):
            if query_lower in trace.user_input.lower():
                results.append(trace.summary())
                if len(results) >= limit:
                    break
        return results


# ── Singleton ────────────────────────────────────────────────────────

_tracer: Optional[TraceRecorder] = None


def get_tracer() -> TraceRecorder:
    """Get the global TraceRecorder instance."""
    global _tracer
    if _tracer is None:
        _tracer = TraceRecorder()
    return _tracer
