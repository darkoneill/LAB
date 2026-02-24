"""
ChromaDB memory backend — wraps the existing VectorStore in the
MemoryBackend interface for seamless swapping with SQLiteBackend.
"""

import json
import logging

from openclaw.memory.backends.base import MemoryBackend

logger = logging.getLogger("openclaw.memory.backends.chromadb")


class ChromaDBBackend(MemoryBackend):
    """MemoryBackend backed by ChromaDB vector search.

    Delegates to :class:`openclaw.memory.vector_store.VectorStore` for
    all heavy lifting (embeddings, cosine similarity, persistence).
    """

    def __init__(self, persist_dir: str | None = None):
        from openclaw.memory.vector_store import VectorStore
        self._store = VectorStore(persist_dir=persist_dir)

    async def store(self, key: str, data: dict) -> None:
        content = data.get("content", "")
        metadata = {
            "category": data.get("category", "general"),
            "significance": data.get("significance", 0.5),
            "access_count": data.get("access_count", 0),
            "created_at": data.get("created_at", 0),
        }
        # ChromaDB metadata must be flat string/int/float/bool
        for k, v in list(metadata.items()):
            if isinstance(v, (dict, list)):
                metadata[k] = json.dumps(v, default=str)
        await self._store.add(content, metadata=metadata, doc_id=key)

    async def recall(self, query: str, top_k: int = 10) -> list[dict]:
        if not query or not query.strip():
            return []
        results = await self._store.search(query, top_k=top_k)
        # Normalise to the shape the rest of the system expects
        normalised = []
        for r in results:
            meta = r.get("metadata", {})
            normalised.append({
                "id": r.get("id", ""),
                "content": r.get("content", ""),
                "category": meta.get("category", "general"),
                "significance": meta.get("significance", 0.5),
                "access_count": meta.get("access_count", 0),
                "created_at": meta.get("created_at", 0),
                "score": r.get("score", 0.0),
                "metadata": meta,
            })
        return normalised

    async def forget(self, key: str) -> None:
        await self._store.delete(key)

    async def count(self) -> int:
        return await self._store.count()

    @property
    def name(self) -> str:
        return "chromadb"
