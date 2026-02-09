"""
OpenClaw Tracing - Observability and Black Box Recorder.
Records every step of the agent pipeline for replay and debugging.
"""

from openclaw.tracing.recorder import TraceRecorder, get_tracer

__all__ = ["TraceRecorder", "get_tracer"]
