"""
MemU-Inspired Memory Manager
Three-layer memory filesystem with autonomous management.

Layer 1 - Resource Layer: Raw data storage (never deleted)
Layer 2 - Item Layer: Extracted memory units
Layer 3 - Category Layer: Aggregated, organized memory files
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

from openclaw.config.settings import get_settings
from .resource_layer import ResourceLayer
from .item_layer import ItemLayer
from .category_layer import CategoryLayer
from .retrieval import HybridRetrieval

logger = logging.getLogger("openclaw.memory")


class MemoryManager:
    """
    Autonomous memory management system.
    Decides what to store, organize, link, evolve, and forget.
    """

    def __init__(self):
        self.settings = get_settings()
        self._base_path = self._resolve_store_path()
        self._base_path.mkdir(parents=True, exist_ok=True)

        self.resources = ResourceLayer(self._base_path / "resources")
        self.items = ItemLayer(self._base_path / "items")
        self.categories = CategoryLayer(self._base_path / "categories")
        self.retrieval = HybridRetrieval(self.items, self.categories)

        self._evolution_task: Optional[asyncio.Task] = None
        self._initialized = False

    def _resolve_store_path(self) -> Path:
        store_path = self.settings.get("memory.store_path", "memory/store")
        p = Path(store_path)
        if p.is_absolute():
            return p
        base = self.settings._base_dir or Path(__file__).parent.parent
        return base / store_path

    async def initialize(self):
        """Load existing memory from disk."""
        if self._initialized:
            return

        await self.resources.load()
        await self.items.load()
        await self.categories.load()
        self._initialized = True

        # Start evolution cycle if enabled
        if self.settings.get("memory.category_layer.evolution.enabled", True):
            self._start_evolution()

        logger.info(
            f"Memory initialized: {self.resources.count} resources, "
            f"{self.items.count} items, {self.categories.count} categories"
        )

    async def store_interaction(self, user_message: str, assistant_response: str, session_id: str = ""):
        """Store a conversation interaction across all layers."""
        if not self.settings.get("memory.enabled", True):
            return

        timestamp = time.time()

        # Layer 1: Store raw resource
        resource_id = await self.resources.store({
            "type": "interaction",
            "user_message": user_message,
            "assistant_response": assistant_response,
            "session_id": session_id,
            "timestamp": timestamp,
        })

        # Layer 2: Extract meaningful items
        items = self._extract_items(user_message, assistant_response, resource_id)
        for item in items:
            await self.items.store(item)

        # Layer 3: Auto-organize into categories
        if self.settings.get("memory.category_layer.auto_organize", True):
            await self.categories.organize(items)

    async def store_knowledge(self, content: str, category: str = "general", metadata: dict = None):
        """Store explicit knowledge."""
        resource_id = await self.resources.store({
            "type": "knowledge",
            "content": content,
            "category": category,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })

        item = {
            "id": f"item_{resource_id}",
            "content": content,
            "category": category,
            "resource_id": resource_id,
            "significance": 0.8,
            "access_count": 1,
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        await self.items.store(item)
        await self.categories.organize([item])

    async def search(self, query: str, top_k: int = 10, method: str = None) -> list[dict]:
        """Search memory using hybrid retrieval."""
        if not self.settings.get("memory.enabled", True):
            return []

        method = method or self.settings.get("memory.retrieval.method", "hybrid")
        return await self.retrieval.search(query, top_k=top_k, method=method)

    async def list_categories(self) -> list[dict]:
        """List all memory categories with summaries."""
        return self.categories.list_all()

    async def get_context_for_prompt(self, query: str, max_items: int = None) -> str:
        """Get formatted memory context for injection into a prompt."""
        max_items = max_items or self.settings.get("memory.category_layer.max_context_items", 50)
        results = await self.search(query, top_k=max_items)

        if not results:
            return ""

        parts = ["The following information is recalled from memory:\n"]
        for r in results:
            content = r.get("content", "")
            category = r.get("category", "general")
            parts.append(f"[{category}] {content}")

        return "\n".join(parts)

    def _extract_items(self, user_msg: str, assistant_msg: str, resource_id: str) -> list[dict]:
        """Extract meaningful memory items from an interaction."""
        items = []
        timestamp = time.time()
        min_significance = self.settings.get("memory.item_layer.min_significance", 0.3)

        # Extract from user message
        user_items = self._analyze_content(user_msg, "user_statement")
        for content, significance in user_items:
            if significance >= min_significance:
                items.append({
                    "content": content,
                    "category": "user_preferences",
                    "resource_id": resource_id,
                    "significance": significance,
                    "access_count": 1,
                    "created_at": timestamp,
                    "source": "user",
                    "metadata": {},
                })

        # Extract from assistant response
        assistant_items = self._analyze_content(assistant_msg, "assistant_knowledge")
        for content, significance in assistant_items:
            if significance >= min_significance:
                items.append({
                    "content": content,
                    "category": "knowledge",
                    "resource_id": resource_id,
                    "significance": significance,
                    "access_count": 1,
                    "created_at": timestamp,
                    "source": "assistant",
                    "metadata": {},
                })

        return items

    def _analyze_content(self, text: str, context_type: str) -> list[tuple[str, float]]:
        """
        Analyze content and extract (text_snippet, significance_score) pairs.
        Uses heuristic analysis since we don't want to call LLM for every message.
        """
        if not text or len(text.strip()) < 10:
            return []

        items = []

        # Split into sentences
        sentences = [s.strip() for s in text.replace("\n", ". ").split(".") if len(s.strip()) > 15]

        # Significance indicators
        high_sig_patterns = [
            "my name is", "i am", "i like", "i prefer", "i want", "i need",
            "remember", "important", "always", "never", "password", "key",
            "je m'appelle", "je suis", "j'aime", "je prefere", "je veux",
            "mon nom", "mon", "ma", "mes", "notre", "nos",
        ]
        medium_sig_patterns = [
            "please", "could you", "how to", "what is", "explain",
            "s'il vous plait", "comment", "qu'est-ce que", "expliquer",
        ]

        for sentence in sentences[:20]:  # Limit processing
            lower = sentence.lower()
            significance = 0.3  # base

            for pattern in high_sig_patterns:
                if pattern in lower:
                    significance = max(significance, 0.8)
                    break

            for pattern in medium_sig_patterns:
                if pattern in lower:
                    significance = max(significance, 0.5)
                    break

            # Longer, more detailed content is more significant
            if len(sentence) > 100:
                significance = min(1.0, significance + 0.1)

            if significance > 0.3:
                items.append((sentence, significance))

        return items

    def _start_evolution(self):
        """Start the background memory evolution cycle."""
        interval = self.settings.get("memory.category_layer.evolution.interval_minutes", 60) * 60

        async def evolution_loop():
            while True:
                await asyncio.sleep(interval)
                try:
                    await self.categories.evolve()
                    if self.settings.get("memory.forgetting.enabled", True):
                        await self._apply_forgetting()
                except Exception as e:
                    logger.error(f"Memory evolution error: {e}")

        try:
            # Use get_running_loop() which is the modern approach (Python 3.10+)
            loop = asyncio.get_running_loop()
            self._evolution_task = loop.create_task(evolution_loop())
        except RuntimeError:
            # No running loop - will be started when initialize() is awaited
            pass

    async def _apply_forgetting(self):
        """Graceful forgetting - reduce priority of unused memories."""
        decay_rate = self.settings.get("memory.forgetting.decay_rate", 0.01)
        min_access = self.settings.get("memory.forgetting.min_access_count", 2)
        grace_days = self.settings.get("memory.forgetting.grace_period_days", 30)
        grace_seconds = grace_days * 86400
        now = time.time()

        for item in self.items.all_items():
            age = now - item.get("created_at", now)
            if age < grace_seconds:
                continue
            if item.get("access_count", 0) < min_access:
                item["significance"] = max(0.05, item.get("significance", 0.5) - decay_rate)

    async def get_stats(self) -> dict:
        return {
            "resources": self.resources.count,
            "items": self.items.count,
            "categories": self.categories.count,
            "store_path": str(self._base_path),
        }
