"""
Abstract base class for memory storage backends.
Backends handle the low-level persistence and retrieval of memory items
(ChromaDB, SQLite FTS5, flat files, etc.).
"""

from abc import ABC, abstractmethod


class MemoryBackend(ABC):
    """Contract that every memory storage backend must fulfil."""

    @abstractmethod
    async def store(self, key: str, data: dict) -> None:
        """Persist a memory item under the given key."""
        ...

    @abstractmethod
    async def recall(self, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve the *top_k* most relevant items for *query*."""
        ...

    @abstractmethod
    async def forget(self, key: str) -> None:
        """Remove a single item by key."""
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return the total number of stored items."""
        ...
