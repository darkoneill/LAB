"""
Tests for openclaw/agent/context.py — ContextManager.
"""

from unittest.mock import MagicMock, patch

import pytest

from openclaw.agent.context import ContextManager


def _make_cm(max_tokens=128000, threshold=0.75, compression_enabled=True):
    """Create a ContextManager with mocked settings."""
    settings = MagicMock()
    settings.get = lambda k, d=None: {
        "agent.context.max_context_tokens": max_tokens,
        "agent.context.compression_threshold": threshold,
        "agent.context.compression_enabled": compression_enabled,
    }.get(k, d)

    with patch("openclaw.agent.context.get_settings", return_value=settings):
        return ContextManager()


def _make_messages(count, content_len=100):
    """Create *count* alternating user/assistant messages."""
    msgs = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": "x" * content_len})
    return msgs


# ── Token counting ──────────────────────────────────────────


class TestTokenCounting:
    def test_estimate_tokens_basic(self):
        cm = _make_cm()
        assert cm.estimate_tokens("hello") == 1  # 5 chars / 4

    def test_estimate_tokens_empty(self):
        cm = _make_cm()
        assert cm.estimate_tokens("") == 0

    def test_estimate_tokens_long_text(self):
        cm = _make_cm()
        text = "a" * 4000
        assert cm.estimate_tokens(text) == 1000

    def test_estimate_messages_tokens(self):
        cm = _make_cm()
        msgs = [
            {"role": "user", "content": "a" * 400},  # 100 tokens + 4 overhead
            {"role": "assistant", "content": "b" * 400},  # 100 tokens + 4 overhead
        ]
        total = cm.estimate_messages_tokens(msgs)
        assert total == 208  # (400/4 + 4) * 2

    def test_estimate_messages_empty_content(self):
        cm = _make_cm()
        msgs = [{"role": "user", "content": ""}]
        assert cm.estimate_messages_tokens(msgs) == 4  # just overhead


# ── Compression detection ───────────────────────────────────


class TestShouldCompress:
    def test_below_threshold_no_compress(self):
        cm = _make_cm(max_tokens=1000, threshold=0.75)
        msgs = _make_messages(2, content_len=40)  # small
        assert cm.should_compress(msgs) is False

    def test_above_threshold_triggers_compress(self):
        cm = _make_cm(max_tokens=100, threshold=0.5)
        # 5 messages × (100/4 + 4) = 5 × 29 = 145 tokens, threshold = 50
        msgs = _make_messages(5, content_len=100)
        assert cm.should_compress(msgs) is True

    def test_compression_disabled(self):
        cm = _make_cm(max_tokens=10, threshold=0.1, compression_enabled=False)
        msgs = _make_messages(100, content_len=1000)
        assert cm.should_compress(msgs) is False

    def test_system_msg_counts_toward_threshold(self):
        cm = _make_cm(max_tokens=100, threshold=0.5)
        msgs = _make_messages(1, content_len=40)
        system_msg = "s" * 200  # 50 tokens from system alone → total > 50
        assert cm.should_compress(msgs, system_msg) is True


# ── Compression ─────────────────────────────────────────────


class TestCompress:
    def test_empty_messages_returned_as_is(self):
        cm = _make_cm()
        assert cm.compress([]) == []

    def test_few_messages_no_summary(self):
        cm = _make_cm(max_tokens=100000)
        msgs = _make_messages(5)
        compressed = cm.compress(msgs)
        # <= 10 messages: returned as-is (all are "recent")
        assert len(compressed) == 5

    def test_many_messages_compressed(self):
        cm = _make_cm(max_tokens=100000)
        msgs = _make_messages(30, content_len=100)
        compressed = cm.compress(msgs)
        # Should have summary + 10 recent messages = 11
        assert len(compressed) == 11
        assert compressed[0]["role"] == "system"
        assert "[Conversation summary:" in compressed[0]["content"]

    def test_summary_contains_user_content(self):
        cm = _make_cm(max_tokens=100000)
        msgs = [
            {"role": "user", "content": "Tell me about quantum computing"},
            {"role": "assistant", "content": "Quantum computing uses qubits"},
        ] * 10  # 20 messages
        compressed = cm.compress(msgs)
        summary = compressed[0]["content"]
        assert "User discussed" in summary

    def test_extreme_recent_overflow(self):
        """When even the 10 most recent messages are too large, keep fewer."""
        cm = _make_cm(max_tokens=200, threshold=0.5)
        # 15 messages of 400 chars each → each ~100 tokens
        msgs = _make_messages(15, content_len=400)
        compressed = cm.compress(msgs, system_tokens=0)
        # Should fall back to keep_recent=5 since 10 × 104 > 100 available
        assert len(compressed) <= 6  # summary + ≤5 recent


# ── build_context ───────────────────────────────────────────


class TestBuildContext:
    def test_build_context_below_threshold(self):
        cm = _make_cm(max_tokens=100000)
        msgs = _make_messages(3)
        sys_msg, out_msgs = cm.build_context(msgs, system_msg="You are helpful.")
        assert sys_msg == "You are helpful."
        assert len(out_msgs) == 3  # Not compressed

    def test_build_context_with_memory(self):
        cm = _make_cm(max_tokens=100000)
        msgs = _make_messages(3)
        sys_msg, out_msgs = cm.build_context(
            msgs, system_msg="Base", memory_context="Remember: user likes Python"
        )
        assert sys_msg == "Base"
        assert len(out_msgs) == 3

    def test_build_context_triggers_compression(self):
        cm = _make_cm(max_tokens=100, threshold=0.3)
        msgs = _make_messages(20, content_len=100)
        sys_msg, out_msgs = cm.build_context(msgs, system_msg="sys")
        # Messages should be compressed
        assert len(out_msgs) < 20

    def test_build_context_with_tools(self):
        cm = _make_cm(max_tokens=100000)
        msgs = _make_messages(2)
        sys_msg, out_msgs = cm.build_context(
            msgs,
            system_msg="System",
            tools_context="Available tools: search, calculate",
        )
        assert sys_msg == "System"
        assert len(out_msgs) == 2


# ── _summarize_messages ─────────────────────────────────────


class TestSummarizeMessages:
    def test_summarize_user_and_assistant(self):
        cm = _make_cm()
        msgs = [
            {"role": "user", "content": "How does Python work?"},
            {"role": "assistant", "content": "Python is an interpreted language."},
        ]
        summary = cm._summarize_messages(msgs)
        assert "User discussed" in summary
        assert "Assistant covered" in summary

    def test_summarize_empty_messages(self):
        cm = _make_cm()
        summary = cm._summarize_messages([])
        assert summary == "Previous conversation context"

    def test_summarize_truncates_long_content(self):
        cm = _make_cm()
        msgs = [{"role": "user", "content": "z" * 500}]
        summary = cm._summarize_messages(msgs)
        # Content should be truncated at 80 chars per message in summary
        assert len(summary) < 500
