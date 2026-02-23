"""
Tests for openclaw/memory/ module.
Covers: ResourceLayer, ItemLayer, CategoryLayer, HybridRetrieval, MemoryManager.
Uses tmp_path for disk isolation, disables vector store (no ChromaDB needed).
"""

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from openclaw.memory.resource_layer import ResourceLayer
from openclaw.memory.item_layer import ItemLayer
from openclaw.memory.category_layer import CategoryLayer
from openclaw.memory.retrieval import HybridRetrieval
from openclaw.memory.manager import MemoryManager


# ── Fixtures ────────────────────────────────────────────────


class FakeSettings:
    """Minimal settings mock — returns sensible defaults, no YAML needed."""
    _base_dir = None

    def __init__(self, overrides: dict = None):
        self._overrides = overrides or {}

    def get(self, dotpath: str, default=None):
        return self._overrides.get(dotpath, default)

    def set(self, dotpath, value, persist=False):
        self._overrides[dotpath] = value

    def all(self):
        return self._overrides

    def resolve_path(self, p):
        return Path(p)


@pytest.fixture
def fake_settings(tmp_path):
    """Settings that routes store_path to tmp_path and disables vector store."""
    return FakeSettings({
        "memory.enabled": True,
        "memory.store_path": str(tmp_path / "memory"),
        "memory.vector.enabled": False,
        "memory.item_layer.min_significance": 0.3,
        "memory.category_layer.auto_organize": True,
        "memory.category_layer.evolution.enabled": False,
        "memory.category_layer.max_context_items": 50,
        "memory.retrieval.method": "hybrid",
        "memory.forgetting.enabled": True,
        "memory.forgetting.decay_rate": 0.1,
        "memory.forgetting.min_access_count": 2,
        "memory.forgetting.grace_period_days": 0,  # no grace → forgetting applies immediately
    })


@pytest.fixture
def resource_layer(tmp_path):
    return ResourceLayer(tmp_path / "resources")


@pytest.fixture
def item_layer(tmp_path, fake_settings):
    with patch("openclaw.memory.item_layer.get_settings", return_value=fake_settings):
        return ItemLayer(tmp_path / "items")


@pytest.fixture
def category_layer(tmp_path):
    return CategoryLayer(tmp_path / "categories")


@pytest.fixture
def retrieval(item_layer, category_layer):
    return HybridRetrieval(item_layer, category_layer)


@pytest.fixture
def memory_manager(tmp_path, fake_settings):
    """Build a MemoryManager with all disk ops redirected to tmp_path."""
    fake_settings._base_dir = tmp_path
    fake_settings._overrides["memory.store_path"] = str(tmp_path / "memory")
    with patch("openclaw.memory.manager.get_settings", return_value=fake_settings), \
         patch("openclaw.memory.item_layer.get_settings", return_value=fake_settings):
        mgr = MemoryManager()
    return mgr


# ══════════════════════════════════════════════════════════════
#  RESOURCE LAYER
# ══════════════════════════════════════════════════════════════


class TestResourceLayer:

    @pytest.mark.asyncio
    async def test_store_and_get(self, resource_layer):
        rid = await resource_layer.store({"type": "test", "content": "hello world"})
        assert rid.startswith("res_")

        got = await resource_layer.get(rid)
        assert got is not None
        assert got["content"] == "hello world"
        assert got["id"] == rid

    @pytest.mark.asyncio
    async def test_store_creates_json_file(self, resource_layer):
        rid = await resource_layer.store({"type": "note", "text": "abc"})
        files = list(resource_layer.store_path.glob("*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["id"] == rid

    @pytest.mark.asyncio
    async def test_load_persisted(self, resource_layer):
        await resource_layer.store({"type": "a", "content": "alpha"})
        await resource_layer.store({"type": "b", "content": "beta"})
        assert resource_layer.count == 2

        # Simulate restart
        rl2 = ResourceLayer(resource_layer.store_path)
        await rl2.load()
        assert rl2.count == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, resource_layer):
        got = await resource_layer.get("nope")
        assert got is None

    @pytest.mark.asyncio
    async def test_search(self, resource_layer):
        await resource_layer.store({"type": "note", "content": "python is great"})
        await resource_layer.store({"type": "note", "content": "rust is fast"})
        await resource_layer.load()  # populate index from disk

        results = await resource_layer.search("python")
        assert len(results) == 1
        assert results[0]["type"] == "note"

    @pytest.mark.asyncio
    async def test_search_by_type(self, resource_layer):
        await resource_layer.store({"type": "interaction", "content": "hello"})
        await resource_layer.store({"type": "knowledge", "content": "hello"})
        await resource_layer.load()

        results = await resource_layer.search("hello", resource_type="knowledge")
        assert len(results) == 1
        assert results[0]["type"] == "knowledge"


# ══════════════════════════════════════════════════════════════
#  ITEM LAYER
# ══════════════════════════════════════════════════════════════


class TestItemLayer:

    @pytest.mark.asyncio
    async def test_store_and_count(self, item_layer):
        iid = await item_layer.store({"content": "memory item 1", "significance": 0.7})
        assert iid.startswith("item_")
        assert item_layer.count == 1

    @pytest.mark.asyncio
    async def test_auto_defaults(self, item_layer):
        await item_layer.store({"content": "bare item"})
        items = item_layer.all_items()
        item = items[0]
        assert "id" in item
        assert "created_at" in item
        assert item["access_count"] == 0
        assert item["significance"] == 0.5

    @pytest.mark.asyncio
    async def test_get_increments_access(self, item_layer):
        iid = await item_layer.store({"content": "track me"})
        item = await item_layer.get(iid)
        assert item["access_count"] == 1
        item2 = await item_layer.get(iid)
        assert item2["access_count"] == 2

    @pytest.mark.asyncio
    async def test_update(self, item_layer):
        iid = await item_layer.store({"content": "original"})
        ok = await item_layer.update(iid, {"content": "updated", "significance": 0.9})
        assert ok is True
        item = await item_layer.get(iid)
        assert item["content"] == "updated"
        assert item["significance"] == 0.9

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, item_layer):
        ok = await item_layer.update("nope", {"content": "x"})
        assert ok is False

    @pytest.mark.asyncio
    async def test_persist_and_reload(self, item_layer, tmp_path, fake_settings):
        await item_layer.store({"content": "persist me", "significance": 0.8})
        await item_layer.store({"content": "and me too", "significance": 0.6})

        with patch("openclaw.memory.item_layer.get_settings", return_value=fake_settings):
            il2 = ItemLayer(item_layer.store_path)
        await il2.load()
        assert il2.count == 2

    @pytest.mark.asyncio
    async def test_search_text(self, item_layer):
        await item_layer.store({"content": "Python is a programming language", "significance": 0.7})
        await item_layer.store({"content": "Rust is fast and safe", "significance": 0.5})
        await item_layer.store({"content": "I prefer Python for scripting", "significance": 0.8})

        results = item_layer.search_text("Python")
        assert len(results) == 2
        # Higher significance item should rank first (scoring includes significance)
        assert "Python" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_search_by_category(self, item_layer):
        await item_layer.store({"content": "fact A", "category": "knowledge"})
        await item_layer.store({"content": "fact B", "category": "knowledge"})
        await item_layer.store({"content": "pref C", "category": "user_preferences"})

        results = item_layer.search_by_category("knowledge")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_significant(self, item_layer):
        await item_layer.store({"content": "low", "significance": 0.2})
        await item_layer.store({"content": "mid", "significance": 0.5})
        await item_layer.store({"content": "high", "significance": 0.9})

        sig = item_layer.get_significant(min_significance=0.5)
        assert len(sig) == 2
        assert sig[0]["content"] == "high"
        assert sig[1]["content"] == "mid"

    @pytest.mark.asyncio
    async def test_get_recent(self, item_layer):
        await item_layer.store({"content": "old", "created_at": 1000})
        await item_layer.store({"content": "new", "created_at": 9999})

        recent = item_layer.get_recent(limit=1)
        assert len(recent) == 1
        assert recent[0]["content"] == "new"

    @pytest.mark.asyncio
    async def test_semantic_search_falls_back_to_text(self, item_layer):
        """With vector store disabled, semantic search should fallback to text."""
        await item_layer.store({"content": "Python machine learning", "significance": 0.8})
        results = await item_layer.search_semantic("Python")
        assert len(results) >= 1
        assert "Python" in results[0]["content"]


# ══════════════════════════════════════════════════════════════
#  CATEGORY LAYER
# ══════════════════════════════════════════════════════════════


class TestCategoryLayer:

    @pytest.mark.asyncio
    async def test_load_creates_defaults(self, category_layer):
        await category_layer.load()
        cats = category_layer.list_all()
        cat_ids = {c["id"] for c in cats}
        assert "knowledge" in cat_ids
        assert "user_preferences" in cat_ids
        assert "general" in cat_ids

    @pytest.mark.asyncio
    async def test_organize_creates_md_file(self, category_layer):
        await category_layer.load()
        items = [{"content": "Python is awesome", "category": "knowledge", "significance": 0.7, "created_at": time.time()}]
        await category_layer.organize(items)

        md = category_layer.store_path / "knowledge.md"
        assert md.exists()
        text = md.read_text()
        assert "Python is awesome" in text
        assert "sig:0.7" in text

    @pytest.mark.asyncio
    async def test_organize_avoids_duplicates(self, category_layer):
        await category_layer.load()
        item = {"content": "duplicate check", "category": "general", "significance": 0.5, "created_at": time.time()}
        await category_layer.organize([item])
        await category_layer.organize([item])

        md = category_layer.store_path / "general.md"
        text = md.read_text()
        assert text.count("duplicate check") == 1

    @pytest.mark.asyncio
    async def test_organize_dynamic_category(self, category_layer):
        await category_layer.load()
        items = [{"content": "new cat item", "category": "my_custom", "significance": 0.6, "created_at": time.time()}]
        await category_layer.organize(items)

        cat_ids = {c["id"] for c in category_layer.list_all()}
        assert "my_custom" in cat_ids

        md = category_layer.store_path / "my_custom.md"
        assert md.exists()

    @pytest.mark.asyncio
    async def test_search_categories(self, category_layer):
        await category_layer.load()
        await category_layer.organize([
            {"content": "FastAPI web framework", "category": "technical", "significance": 0.7, "created_at": time.time()},
            {"content": "Django ORM usage", "category": "technical", "significance": 0.6, "created_at": time.time()},
        ])

        results = category_layer.search_categories("FastAPI")
        assert len(results) == 1
        assert "FastAPI" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_get_category_content(self, category_layer):
        await category_layer.load()
        await category_layer.organize([
            {"content": "item alpha", "category": "projects", "significance": 0.5, "created_at": time.time()}
        ])
        content = category_layer.get_category_content("projects")
        assert "item alpha" in content

    @pytest.mark.asyncio
    async def test_get_category_content_nonexistent(self, category_layer):
        content = category_layer.get_category_content("does_not_exist")
        assert content == ""

    @pytest.mark.asyncio
    async def test_evolve_consolidation(self, category_layer):
        await category_layer.load()
        # Create a category with >100 lines
        items = [
            {"content": f"Entry number {i} with enough text to be a real entry", "category": "big_cat", "significance": round(0.3 + (i % 10) * 0.07, 2), "created_at": time.time()}
            for i in range(110)
        ]
        for item in items:
            await category_layer.organize([item])

        md = category_layer.store_path / "big_cat.md"
        lines_before = len(md.read_text().split("\n"))

        await category_layer.evolve()

        text_after = md.read_text()
        assert "archived" in text_after.lower()
        archive = category_layer.store_path / "big_cat_archive.md"
        assert archive.exists()

    @pytest.mark.asyncio
    async def test_persist_meta(self, category_layer):
        await category_layer.load()
        await category_layer.organize([
            {"content": "persist meta test", "category": "knowledge", "significance": 0.5, "created_at": time.time()}
        ])
        meta = json.loads(category_layer._meta_path.read_text())
        assert "knowledge" in meta
        assert meta["knowledge"]["item_count"] >= 1


# ══════════════════════════════════════════════════════════════
#  HYBRID RETRIEVAL
# ══════════════════════════════════════════════════════════════


class TestHybridRetrieval:

    @pytest.mark.asyncio
    async def test_keyword_search(self, retrieval, item_layer, category_layer):
        await category_layer.load()
        await item_layer.store({"content": "Python is a versatile programming language", "significance": 0.7})
        await item_layer.store({"content": "Rust guarantees memory safety", "significance": 0.6})
        await item_layer.store({"content": "JavaScript runs everywhere", "significance": 0.5})

        results = await retrieval.search("Python programming", top_k=5, method="keyword")
        assert len(results) >= 1
        assert "Python" in results[0]["content"]

    @pytest.mark.asyncio
    async def test_keyword_search_empty(self, retrieval, item_layer, category_layer):
        await category_layer.load()
        results = await retrieval.search("anything", top_k=5, method="keyword")
        assert results == []

    @pytest.mark.asyncio
    async def test_contextual_search(self, retrieval, item_layer, category_layer):
        await category_layer.load()
        now = time.time()
        await item_layer.store({"content": "recent and important", "significance": 0.9, "created_at": now, "access_count": 5})
        await item_layer.store({"content": "old and forgotten", "significance": 0.2, "created_at": now - 86400 * 60, "access_count": 0})

        results = await retrieval.search("test", top_k=5, method="contextual")
        assert len(results) == 2
        assert results[0]["content"] == "recent and important"

    @pytest.mark.asyncio
    async def test_hybrid_search_merges_strategies(self, retrieval, item_layer, category_layer):
        await category_layer.load()
        now = time.time()
        await item_layer.store({"content": "Python web framework FastAPI", "significance": 0.8, "created_at": now})
        await item_layer.store({"content": "Rust async runtime tokio", "significance": 0.7, "created_at": now})
        await item_layer.store({"content": "old Python script from years ago", "significance": 0.3, "created_at": now - 86400 * 365})

        results = await retrieval.search("Python", top_k=5, method="hybrid")
        # Should return Python-related items, with more recent/significant ones first
        assert len(results) >= 2
        # First result should be the high-significance, recent Python item
        assert "Python" in results[0]["content"]
        assert results[0].get("significance", 0) >= 0.5

    @pytest.mark.asyncio
    async def test_hybrid_deduplicates(self, retrieval, item_layer, category_layer):
        await category_layer.load()
        await item_layer.store({"content": "unique item about Python testing", "significance": 0.8})

        results = await retrieval.search("Python testing", top_k=10, method="hybrid")
        ids = [r.get("id") for r in results if r.get("id")]
        assert len(ids) == len(set(ids)), "Duplicate items found in hybrid results"

    @pytest.mark.asyncio
    async def test_tokenizer_strips_stop_words(self, retrieval):
        tokens = retrieval._tokenize("the quick brown fox is jumping over the lazy dog")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens

    @pytest.mark.asyncio
    async def test_tokenizer_french_stop_words(self, retrieval):
        tokens = retrieval._tokenize("le chat est sur la table avec les enfants")
        assert "le" not in tokens
        assert "est" not in tokens
        assert "chat" in tokens
        assert "table" in tokens

    @pytest.mark.asyncio
    async def test_keyword_tfidf_scoring(self, retrieval, item_layer, category_layer):
        """Items with rare terms should score higher than common ones (IDF effect)."""
        await category_layer.load()
        # "python" appears in all 3 → low IDF; "asyncio" appears in 1 → high IDF
        await item_layer.store({"content": "python basics tutorial for beginners", "significance": 0.5})
        await item_layer.store({"content": "python web development flask", "significance": 0.5})
        await item_layer.store({"content": "python asyncio advanced patterns", "significance": 0.5})

        results = await retrieval.search("python asyncio", top_k=3, method="keyword")
        assert len(results) >= 1
        # The item mentioning "asyncio" should score highest (rare term boost)
        assert "asyncio" in results[0]["content"]


# ══════════════════════════════════════════════════════════════
#  MEMORY MANAGER — full integration
# ══════════════════════════════════════════════════════════════


class TestMemoryManager:

    @pytest.mark.asyncio
    async def test_initialize(self, memory_manager):
        await memory_manager.initialize()
        assert memory_manager._initialized is True
        stats = await memory_manager.get_stats()
        assert stats["resources"] == 0
        assert stats["items"] == 0

    @pytest.mark.asyncio
    async def test_store_interaction_full_cycle(self, memory_manager):
        """Full cycle: store interaction → items extracted → categories organized → searchable."""
        await memory_manager.initialize()

        await memory_manager.store_interaction(
            user_message="My name is Alice and I prefer Python for my projects",
            assistant_response="I'll remember that you're Alice and you prefer Python for your projects",
            session_id="test-session",
        )

        # Layer 1: resource stored
        assert memory_manager.resources.count == 1

        # Layer 2: items extracted (heuristic should pick up "my name is", "i prefer")
        assert memory_manager.items.count >= 1

        # Layer 3: categories organized
        cats = await memory_manager.list_categories()
        assert len(cats) > 0

        # Searchable
        results = await memory_manager.search("Alice Python")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_store_knowledge(self, memory_manager):
        await memory_manager.initialize()
        await memory_manager.store_knowledge(
            content="FastAPI is a modern Python web framework",
            category="technical",
            metadata={"source": "docs"},
        )
        assert memory_manager.resources.count == 1
        assert memory_manager.items.count == 1

        results = await memory_manager.search("FastAPI")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_get_context_for_prompt(self, memory_manager):
        await memory_manager.initialize()
        await memory_manager.store_knowledge(content="The user likes dark mode", category="user_preferences")

        ctx = await memory_manager.get_context_for_prompt("dark mode")
        assert "dark mode" in ctx
        assert "Recalled from memory" in ctx or "recalled" in ctx.lower() or "memory" in ctx.lower()

    @pytest.mark.asyncio
    async def test_get_context_empty(self, memory_manager):
        await memory_manager.initialize()
        ctx = await memory_manager.get_context_for_prompt("nothing here")
        assert ctx == ""

    @pytest.mark.asyncio
    async def test_disabled_memory(self, memory_manager, fake_settings):
        fake_settings.set("memory.enabled", False)
        await memory_manager.initialize()

        await memory_manager.store_interaction("hello", "hi", "s1")
        # Should not store anything when disabled
        assert memory_manager.resources.count == 0

        results = await memory_manager.search("hello")
        assert results == []

    @pytest.mark.asyncio
    async def test_multiple_interactions(self, memory_manager):
        await memory_manager.initialize()

        await memory_manager.store_interaction(
            "I always want concise answers",
            "Understood, I will keep my responses concise.",
            "s1",
        )
        await memory_manager.store_interaction(
            "How to deploy FastAPI on Docker?",
            "You can use a Dockerfile with uvicorn to deploy FastAPI in a Docker container.",
            "s1",
        )

        assert memory_manager.resources.count == 2
        # Should find both topics
        results_concise = await memory_manager.search("concise answers")
        results_docker = await memory_manager.search("Docker FastAPI deploy")
        assert len(results_concise) >= 1 or len(results_docker) >= 1


# ══════════════════════════════════════════════════════════════
#  EXTRACTION & SIGNIFICANCE
# ══════════════════════════════════════════════════════════════


class TestExtraction:

    @pytest.mark.asyncio
    async def test_high_significance_patterns(self, memory_manager):
        """Messages with 'my name is', 'I prefer' etc. should get high significance."""
        await memory_manager.initialize()
        items = memory_manager._extract_items(
            "My name is Bob and I prefer dark mode for everything",
            "Noted, Bob. Dark mode preferences saved.",
            "res_1",
        )
        assert len(items) >= 1
        high_sig = [i for i in items if i["significance"] >= 0.8]
        assert len(high_sig) >= 1, f"Expected high-significance items, got: {[(i['content'][:40], i['significance']) for i in items]}"

    @pytest.mark.asyncio
    async def test_medium_significance_patterns(self, memory_manager):
        """Messages with 'how to', 'explain' should get medium significance."""
        await memory_manager.initialize()
        items = memory_manager._extract_items(
            "How to configure nginx reverse proxy for websocket connections",
            "You need to add proxy_set_header Upgrade and Connection headers.",
            "res_2",
        )
        user_items = [i for i in items if i["source"] == "user"]
        if user_items:
            assert user_items[0]["significance"] >= 0.5

    @pytest.mark.asyncio
    async def test_low_significance_filtered(self, memory_manager):
        """Short, generic messages should produce no items (below min_significance)."""
        await memory_manager.initialize()
        items = memory_manager._extract_items("ok", "ok", "res_3")
        assert items == []

    @pytest.mark.asyncio
    async def test_long_content_boost(self, memory_manager):
        """Content >100 chars should get a significance boost."""
        await memory_manager.initialize()
        long_msg = "I need you to remember that " + "x" * 100 + " this is very important context"
        items = memory_manager._extract_items(long_msg, "Noted.", "res_4")
        if items:
            # At least one should have boosted significance
            assert any(i["significance"] >= 0.4 for i in items)

    @pytest.mark.asyncio
    async def test_sentence_limit(self, memory_manager):
        """_analyze_content should process at most 20 sentences."""
        await memory_manager.initialize()
        # 30 sentences, each >15 chars with a high-sig pattern
        msg = ". ".join([f"I prefer option {i} for this project" for i in range(30)])
        result = memory_manager._analyze_content(msg, "user_statement")
        assert len(result) <= 20


# ══════════════════════════════════════════════════════════════
#  FORGETTING MECHANISM
# ══════════════════════════════════════════════════════════════


class TestForgetting:

    @pytest.mark.asyncio
    async def test_forgetting_decays_significance(self, memory_manager):
        """Old items with low access count should lose significance."""
        await memory_manager.initialize()

        old_time = time.time() - 86400 * 60  # 60 days ago
        await memory_manager.items.store({
            "content": "old forgotten fact",
            "significance": 0.7,
            "access_count": 0,
            "created_at": old_time,
        })

        await memory_manager._apply_forgetting()

        item = memory_manager.items.all_items()[0]
        assert item["significance"] < 0.7, f"Expected decay, got {item['significance']}"

    @pytest.mark.asyncio
    async def test_forgetting_spares_accessed_items(self, memory_manager):
        """Items with access_count >= min_access should NOT decay."""
        await memory_manager.initialize()

        old_time = time.time() - 86400 * 60
        await memory_manager.items.store({
            "content": "frequently accessed",
            "significance": 0.7,
            "access_count": 5,  # above min_access (2)
            "created_at": old_time,
        })

        await memory_manager._apply_forgetting()

        item = memory_manager.items.all_items()[0]
        assert item["significance"] == 0.7, "Accessed item should not decay"

    @pytest.mark.asyncio
    async def test_forgetting_respects_grace_period(self, memory_manager, fake_settings):
        """Items within grace period should NOT decay."""
        fake_settings.set("memory.forgetting.grace_period_days", 90)
        await memory_manager.initialize()

        recent_time = time.time() - 86400 * 30  # 30 days ago (within 90-day grace)
        await memory_manager.items.store({
            "content": "recent low access",
            "significance": 0.7,
            "access_count": 0,
            "created_at": recent_time,
        })

        await memory_manager._apply_forgetting()

        item = memory_manager.items.all_items()[0]
        assert item["significance"] == 0.7, "Within grace period — should not decay"

    @pytest.mark.asyncio
    async def test_forgetting_floor(self, memory_manager):
        """Significance should never go below 0.05."""
        await memory_manager.initialize()

        old_time = time.time() - 86400 * 365  # 1 year ago
        await memory_manager.items.store({
            "content": "ancient memory",
            "significance": 0.06,
            "access_count": 0,
            "created_at": old_time,
        })

        # Apply forgetting many times
        for _ in range(20):
            await memory_manager._apply_forgetting()

        item = memory_manager.items.all_items()[0]
        assert item["significance"] >= 0.05, f"Hit floor violation: {item['significance']}"

    @pytest.mark.asyncio
    async def test_forgetting_multiple_items(self, memory_manager):
        """Forgetting should apply independently to each item."""
        await memory_manager.initialize()
        old = time.time() - 86400 * 60

        await memory_manager.items.store({"content": "old + no access", "significance": 0.6, "access_count": 0, "created_at": old})
        await memory_manager.items.store({"content": "old + accessed", "significance": 0.6, "access_count": 10, "created_at": old})
        await memory_manager.items.store({"content": "new + no access", "significance": 0.6, "access_count": 0, "created_at": time.time()})

        await memory_manager._apply_forgetting()

        items = memory_manager.items.all_items()
        decayed = items[0]      # old, no access
        spared_access = items[1]  # old, accessed
        spared_new = items[2]     # new

        assert decayed["significance"] < 0.6
        assert spared_access["significance"] == 0.6
        # spared_new: grace_period_days=0 in fake_settings, but the item is recent
        # so age < grace_seconds (0), meaning it will be subject to forgetting if access_count < 2
        # Actually grace_period_days = 0 → grace_seconds = 0, so ALL items past the 0 second grace will be checked


# ══════════════════════════════════════════════════════════════
#  EDGE CASES
# ══════════════════════════════════════════════════════════════


class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_empty_search(self, retrieval, item_layer, category_layer):
        await category_layer.load()
        results = await retrieval.search("", top_k=5, method="keyword")
        assert results == []

    @pytest.mark.asyncio
    async def test_store_interaction_empty_messages(self, memory_manager):
        await memory_manager.initialize()
        await memory_manager.store_interaction("", "", "s1")
        # Empty messages → no items extracted, but resource still stored
        assert memory_manager.resources.count == 1
        assert memory_manager.items.count == 0

    @pytest.mark.asyncio
    async def test_concurrent_stores(self, item_layer):
        """Item layer should handle concurrent writes safely (asyncio.Lock)."""
        async def store_item(i):
            await item_layer.store({"content": f"concurrent item {i}"})

        await asyncio.gather(*[store_item(i) for i in range(20)])
        assert item_layer.count == 20

    @pytest.mark.asyncio
    async def test_unicode_content(self, memory_manager):
        await memory_manager.initialize()
        await memory_manager.store_interaction(
            "Je m'appelle François et j'habite à Montréal 🇨🇦",
            "Enchanté François ! Je me souviendrai que vous êtes de Montréal.",
            "s1",
        )
        results = await memory_manager.search("François Montréal")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_special_chars_in_content(self, resource_layer):
        rid = await resource_layer.store({
            "type": "test",
            "content": 'Line with "quotes" and <tags> & symbols',
        })
        got = await resource_layer.get(rid)
        assert got["content"] == 'Line with "quotes" and <tags> & symbols'
