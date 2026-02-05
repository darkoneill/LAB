"""
Context Manager - Dynamic context compression and management.
Inspired by AgentZero's context engineering.
"""

import logging
from typing import Optional

from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.agent.context")


class ContextManager:
    """
    Manages conversation context with:
    - Token counting (approximate)
    - Dynamic compression when context grows too large
    - Priority-based message retention
    - Memory integration
    """

    # Approximate tokens per character for different languages
    CHARS_PER_TOKEN = 4  # English average

    def __init__(self):
        self.settings = get_settings()
        self.max_tokens = self.settings.get("agent.context.max_context_tokens", 128000)
        self.compression_threshold = self.settings.get("agent.context.compression_threshold", 0.75)

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation."""
        return len(text) // self.CHARS_PER_TOKEN

    def estimate_messages_tokens(self, messages: list[dict]) -> int:
        """Estimate total tokens in a message list."""
        total = 0
        for msg in messages:
            total += self.estimate_tokens(msg.get("content", ""))
            total += 4  # message overhead
        return total

    def should_compress(self, messages: list[dict], system_msg: str = "") -> bool:
        """Check if context should be compressed."""
        if not self.settings.get("agent.context.compression_enabled", True):
            return False
        total = self.estimate_tokens(system_msg) + self.estimate_messages_tokens(messages)
        return total > (self.max_tokens * self.compression_threshold)

    def compress(self, messages: list[dict], system_tokens: int = 0) -> list[dict]:
        """
        Compress context by:
        1. Keeping the most recent messages
        2. Summarizing older messages
        3. Preserving system-critical messages
        """
        if not messages:
            return messages

        available_tokens = int(self.max_tokens * self.compression_threshold) - system_tokens

        # Always keep the last N messages
        keep_recent = 10
        recent = messages[-keep_recent:] if len(messages) > keep_recent else messages
        recent_tokens = self.estimate_messages_tokens(recent)

        if recent_tokens >= available_tokens:
            # Even recent messages are too many, keep fewer
            keep_recent = max(4, keep_recent // 2)
            recent = messages[-keep_recent:]

        older = messages[:-keep_recent] if len(messages) > keep_recent else []

        if not older:
            return recent

        # Summarize older messages
        summary = self._summarize_messages(older)
        compressed = [{"role": "system", "content": f"[Conversation summary: {summary}]"}]
        compressed.extend(recent)

        return compressed

    def _summarize_messages(self, messages: list[dict]) -> str:
        """Create a brief summary of messages."""
        topics = set()
        user_msgs = []
        assistant_msgs = []

        for msg in messages:
            content = msg.get("content", "")[:200]
            role = msg.get("role", "user")
            if role == "user":
                user_msgs.append(content)
            elif role == "assistant":
                assistant_msgs.append(content)

        # Simple extractive summary
        summary_parts = []
        if user_msgs:
            summary_parts.append(f"User discussed: {'; '.join(m[:80] for m in user_msgs[-5:])}")
        if assistant_msgs:
            summary_parts.append(f"Assistant covered: {'; '.join(m[:80] for m in assistant_msgs[-5:])}")

        return " | ".join(summary_parts) if summary_parts else "Previous conversation context"

    def build_context(
        self,
        messages: list[dict],
        system_msg: str,
        memory_context: str = "",
        tools_context: str = "",
    ) -> tuple[str, list[dict]]:
        """
        Build the full context for an LLM call.
        Returns (system_message, formatted_messages).
        """
        system_tokens = self.estimate_tokens(system_msg + memory_context + tools_context)

        if self.should_compress(messages, system_msg):
            messages = self.compress(messages, system_tokens)
            logger.info(f"Context compressed to {len(messages)} messages")

        return system_msg, messages
