"""
Item Layer - Extracted memory units with vector embeddings.
Each item is the smallest meaningful unit that can be understood independently.
Enhanced with ChromaDB integration for semantic search (RAG).
"""

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.memory.items")


class ItemLayer:
    """
    Stores discrete memory items extracted from resources.
    Items are the building blocks for categories and retrieval.
    Thread-safe with asyncio.Lock for concurrent access.

    Enhanced with:
    - Vector embeddings via ChromaDB
    - Semantic similarity search
    - Hybrid retrieval (keyword + semantic)
    """

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._items: list[dict] = []
        self._index_path = self.store_path / "_index.json"
        self._lock = asyncio.Lock()
        self._vector_store = None
        self.settings = get_settings()

    async def _get_vector_store(self):
        """Lazy initialization of vector store."""
        if self._vector_store is None and self.settings.get("memory.vector.enabled", True):
            try:
                from openclaw.memory.vector_store import VectorStore
                self._vector_store = VectorStore(
                    persist_dir=str(self.store_path / "vectors")
                )
                await self._vector_store.initialize()
            except ImportError:
                logger.warning("ChromaDB not available, using text-only search")
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
        return self._vector_store

    async def load(self):
        """Load items from disk."""
        self._items.clear()
        if self._index_path.exists():
            try:
                data = json.loads(self._index_path.read_text(encoding="utf-8"))
                self._items = data if isinstance(data, list) else []
            except Exception as e:
                logger.warning(f"Failed to load items index: {e}")

        logger.info(f"Loaded {len(self._items)} memory items")

    async def store(self, item: dict) -> str:
        """Store a new memory item with vector embedding."""
        async with self._lock:
            if "id" not in item:
                item["id"] = f"item_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            if "created_at" not in item:
                item["created_at"] = time.time()
            if "access_count" not in item:
                item["access_count"] = 0
            if "significance" not in item:
                item["significance"] = 0.5
            if "last_accessed" not in item:
                item["last_accessed"] = time.time()

            self._items.append(item)
            await self._persist_unlocked()

        # Add to vector store (outside lock to avoid blocking)
        vector_store = await self._get_vector_store()
        if vector_store and item.get("content"):
            try:
                await vector_store.add(
                    content=item["content"],
                    metadata={
                        "item_id": item["id"],
                        "category": item.get("category", ""),
                        "significance": item.get("significance", 0.5),
                        "created_at": item.get("created_at", 0),
                    },
                    doc_id=item["id"]
                )
            except Exception as e:
                logger.warning(f"Failed to add item to vector store: {e}")

        return item["id"]

    async def update(self, item_id: str, updates: dict):
        """Update an existing item."""
        async with self._lock:
            for item in self._items:
                if item.get("id") == item_id:
                    item.update(updates)
                    await self._persist_unlocked()
                    return True
            return False

    async def get(self, item_id: str) -> Optional[dict]:
        """Get an item by ID and increment access count."""
        for item in self._items:
            if item.get("id") == item_id:
                item["access_count"] = item.get("access_count", 0) + 1
                item["last_accessed"] = time.time()
                return item
        return None

    def search_text(self, query: str, limit: int = 20) -> list[dict]:
        """Simple text search across items."""
        query_lower = query.lower()
        results = []
        for item in self._items:
            content = item.get("content", "").lower()
            if query_lower in content:
                score = self._compute_relevance(item, query_lower)
                results.append({**item, "_score": score})

        results.sort(key=lambda x: x.get("_score", 0), reverse=True)
        return results[:limit]

    async def search_semantic(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic similarity search using vector embeddings."""
        vector_store = await self._get_vector_store()
        if not vector_store:
            # Fallback to text search
            return self.search_text(query, limit)

        try:
            results = await vector_store.search(query, top_k=limit)
            # Enrich with full item data
            enriched = []
            for r in results:
                item_id = r.get("id") or r.get("metadata", {}).get("item_id")
                item = await self.get(item_id) if item_id else None
                if item:
                    enriched.append({
                        **item,
                        "_score": r.get("score", 0),
                        "_semantic": True,
                    })
                else:
                    # Item from vector store but not in index
                    enriched.append({
                        "id": item_id,
                        "content": r.get("content", ""),
                        "_score": r.get("score", 0),
                        "_semantic": True,
                    })
            return enriched
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return self.search_text(query, limit)

    async def search_hybrid(self, query: str, limit: int = 10) -> list[dict]:
        """
        Hybrid search combining semantic and keyword matching.
        Provides best of both worlds for RAG.
        """
        # Get results from both methods
        text_results = self.search_text(query, limit * 2)
        semantic_results = await self.search_semantic(query, limit * 2)

        # Merge and deduplicate by ID
        seen_ids = set()
        merged = []

        # Process semantic results first (usually more relevant)
        for r in semantic_results:
            item_id = r.get("id")
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                # Boost score for semantic matches
                r["_score"] = r.get("_score", 0) * 1.2
                merged.append(r)

        # Add text results that weren't in semantic
        for r in text_results:
            item_id = r.get("id")
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                merged.append(r)

        # Sort by score and return top results
        merged.sort(key=lambda x: x.get("_score", 0), reverse=True)
        return merged[:limit]

    def search_by_category(self, category: str, limit: int = 20) -> list[dict]:
        """Get items in a specific category."""
        results = [i for i in self._items if i.get("category") == category]
        results.sort(key=lambda x: x.get("significance", 0), reverse=True)
        return results[:limit]

    def all_items(self) -> list[dict]:
        """Return all items."""
        return self._items

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Get most recently created items."""
        sorted_items = sorted(self._items, key=lambda x: x.get("created_at", 0), reverse=True)
        return sorted_items[:limit]

    def get_significant(self, min_significance: float = 0.5, limit: int = 20) -> list[dict]:
        """Get most significant items."""
        filtered = [i for i in self._items if i.get("significance", 0) >= min_significance]
        filtered.sort(key=lambda x: x.get("significance", 0), reverse=True)
        return filtered[:limit]

    def _compute_relevance(self, item: dict, query: str) -> float:
        """Compute relevance score for an item against a query."""
        content = item.get("content", "").lower()
        significance = item.get("significance", 0.5)
        access_count = item.get("access_count", 0)

        # Text match score
        words = query.split()
        match_count = sum(1 for w in words if w in content)
        text_score = match_count / max(len(words), 1)

        # Recency score
        age_hours = (time.time() - item.get("created_at", time.time())) / 3600
        recency_score = max(0, 1.0 - (age_hours / 720))  # Decay over 30 days

        # Combined score
        return (text_score * 0.4) + (significance * 0.3) + (recency_score * 0.2) + (min(access_count / 10, 1) * 0.1)

    async def _persist_unlocked(self):
        """Save items to disk (caller must hold lock)."""
        self._index_path.write_text(
            json.dumps(self._items, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    async def _persist(self):
        """Save items to disk with lock."""
        async with self._lock:
            await self._persist_unlocked()

    @property
    def count(self) -> int:
        return len(self._items)
