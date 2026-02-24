"""
Abstract base class for communication channels.
A channel is a transport that connects users to the assistant
(terminal, web UI, Telegram, Slack, etc.).
"""

from abc import ABC, abstractmethod


class ChannelBase(ABC):
    """Contract for all communication channels."""

    @abstractmethod
    async def start(self) -> None:
        """Start listening on this channel."""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the channel."""
        ...

    @abstractmethod
    async def send_message(self, recipient: str, content: str) -> None:
        """Send a message to a specific recipient/session."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable channel name (e.g. 'terminal', 'telegram')."""
        ...
