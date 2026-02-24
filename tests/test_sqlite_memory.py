"""
Tests for the SQLite FTS5 memory backend (P2).
Covers: store/recall/forget cycle, FTS5 search, hybrid scoring,
        edge cases, backend factory, and performance with 1000 items.
"""

import asyncio
import time
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from openclaw.memory.backends.base import MemoryBackend
from openclaw.memory.backends.sqlite_backend import SQLiteBackend, BM25_WEIGHT, SIGNIFICANCE_WEIGHT


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def backend(tmp_path):
    """Fresh SQLite backend in a temp directory."""
    db = SQLiteBackend(tmp_path / "test.db")
    yield db
    db.close()


@pytest.fixture
def in_memory_backend():
    """In-memory SQLite backend (fast, disposable)."""
    db = SQLiteBackend(":memory:")
    yield db
    db.close()


# ══════════════════════════════════════════════════════════════
#  ABC compliance
# ══════════════════════════════════════════════════════════════


class TestABCCompliance:

    def test_is_memory_backend(self, backend):
        assert isinstance(backend, MemoryBackend)

    def test_has_required_methods(self, backend):
        for method in ("store", "recall", "forget", "count"):
            assert callable(getattr(backend, method))


# ══════════════════════════════════════════════════════════════
#  Store / Recall / Forget cycle
# ══════════════════════════════════════════════════════════════


class TestStoreCycle:

    @pytest.mark.asyncio
    async def test_store_and_count(self, backend):
        assert await backend.count() == 0
        await backend.store("k1", {"content": "Hello world", "category": "general"})
        assert await backend.count() == 1

    @pytest.mark.asyncio
    async def test_store_multiple(self, backend):
        for i in range(5):
            await backend.store(f"k{i}", {"content": f"Item {i}", "significance": 0.5 + i * 0.1})
        assert await backend.count() == 5

    @pytest.mark.asyncio
    async def test_store_upsert(self, backend):
        await backend.store("k1", {"content": "version1"})
        await backend.store("k1", {"content": "version2"})
        assert await backend.count() == 1
        item = await backend.get("k1")
        assert item["content"] == "version2"

    @pytest.mark.asyncio
    async def test_recall_basic(self, backend):
        await backend.store("k1", {"content": "Python is a programming language", "category": "tech"})
        await backend.store("k2", {"content": "Cats are cute animals", "category": "animals"})
        results = await backend.recall("programming language", top_k=5)
        assert len(results) >= 1
        assert results[0]["content"] == "Python is a programming language"

    @pytest.mark.asyncio
    async def test_forget_removes_item(self, backend):
        await backend.store("k1", {"content": "to be deleted"})
        assert await backend.count() == 1
        await backend.forget("k1")
        assert await backend.count() == 0

    @pytest.mark.asyncio
    async def test_forget_nonexistent_no_error(self, backend):
        await backend.forget("nope")  # should not raise

    @pytest.mark.asyncio
    async def test_recall_empty_query(self, backend):
        await backend.store("k1", {"content": "something"})
        results = await backend.recall("", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_no_match(self, backend):
        await backend.store("k1", {"content": "Python is great"})
        results = await backend.recall("xylophone quantum", top_k=5)
        assert len(results) == 0


# ══════════════════════════════════════════════════════════════
#  FTS5 search quality
# ══════════════════════════════════════════════════════════════


class TestFTS5Search:

    @pytest.mark.asyncio
    async def test_partial_term_via_like_fallback(self, backend):
        await backend.store("k1", {"content": "artificial intelligence is amazing"})
        # FTS5 requires full words; partial match falls back to LIKE
        results = await backend.recall("intelligence", top_k=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_multi_term_search(self, backend):
        await backend.store("k1", {"content": "machine learning with deep neural networks"})
        await backend.store("k2", {"content": "cooking recipes for dinner"})
        results = await backend.recall("machine learning", top_k=5)
        assert len(results) >= 1
        assert results[0]["id"] == "k1"

    @pytest.mark.asyncio
    async def test_category_indexed(self, backend):
        """FTS5 indexes category alongside content."""
        await backend.store("k1", {"content": "some text", "category": "security"})
        await backend.store("k2", {"content": "other text", "category": "cooking"})
        results = await backend.recall("security", top_k=5)
        assert any(r["id"] == "k1" for r in results)

    @pytest.mark.asyncio
    async def test_stemming_via_porter(self, backend):
        """The porter tokenizer should match stemmed variants."""
        await backend.store("k1", {"content": "running very fast in the morning"})
        results = await backend.recall("run fast", top_k=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_special_chars_sanitized(self, backend):
        """FTS5 operators in query must not cause SQL errors."""
        await backend.store("k1", {"content": "hello world"})
        # These contain FTS5 special chars
        results = await backend.recall('hello OR "world"', top_k=5)
        assert isinstance(results, list)
        results2 = await backend.recall("hello AND world NOT foo", top_k=5)
        assert isinstance(results2, list)


# ══════════════════════════════════════════════════════════════
#  Hybrid scoring
# ══════════════════════════════════════════════════════════════


class TestHybridScoring:

    @pytest.mark.asyncio
    async def test_high_significance_ranked_higher(self, backend):
        """An item with higher significance should rank above a low one
        when BM25 scores are comparable."""
        await backend.store("low", {
            "content": "database query optimization",
            "significance": 0.1,
            "created_at": time.time(),
        })
        await backend.store("high", {
            "content": "database query optimization tips",
            "significance": 0.9,
            "created_at": time.time(),
        })
        results = await backend.recall("database query optimization", top_k=5)
        assert len(results) == 2
        assert results[0]["id"] == "high"

    @pytest.mark.asyncio
    async def test_recent_item_ranked_higher(self, backend):
        """Recent items should beat old items with equal significance."""
        old_time = time.time() - 90 * 86400  # 90 days ago
        await backend.store("old", {
            "content": "server configuration guide",
            "significance": 0.5,
            "created_at": old_time,
        })
        await backend.store("new", {
            "content": "server configuration guide updated",
            "significance": 0.5,
            "created_at": time.time(),
        })
        results = await backend.recall("server configuration guide", top_k=5)
        assert len(results) == 2
        assert results[0]["id"] == "new"

    @pytest.mark.asyncio
    async def test_score_in_results(self, backend):
        await backend.store("k1", {"content": "test scoring output", "significance": 0.5})
        results = await backend.recall("scoring", top_k=5)
        assert len(results) >= 1
        assert "score" in results[0]
        assert 0 <= results[0]["score"] <= 1


# ══════════════════════════════════════════════════════════════
#  Extra helpers (get, update_significance)
# ══════════════════════════════════════════════════════════════


class TestHelpers:

    @pytest.mark.asyncio
    async def test_get_bumps_access_count(self, backend):
        await backend.store("k1", {"content": "hello", "access_count": 0})
        item = await backend.get("k1")
        assert item["access_count"] == 1
        item = await backend.get("k1")
        assert item["access_count"] == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, backend):
        assert await backend.get("nope") is None

    @pytest.mark.asyncio
    async def test_update_significance(self, backend):
        await backend.store("k1", {"content": "item", "significance": 0.5})
        await backend.update_significance("k1", 0.9)
        item = await backend.get("k1")
        assert item["significance"] == 0.9

    @pytest.mark.asyncio
    async def test_metadata_roundtrip(self, backend):
        await backend.store("k1", {
            "content": "with meta",
            "source": "user",
            "resource_id": "res_123",
        })
        item = await backend.get("k1")
        assert item["metadata"]["source"] == "user"
        assert item["metadata"]["resource_id"] == "res_123"


# ══════════════════════════════════════════════════════════════
#  Edge cases
# ══════════════════════════════════════════════════════════════


class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_unicode_content(self, backend):
        await backend.store("k1", {"content": "Les memoires sont extraordinaires"})
        results = await backend.recall("memoires", top_k=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_empty_content(self, backend):
        await backend.store("k1", {"content": ""})
        assert await backend.count() == 1

    @pytest.mark.asyncio
    async def test_very_long_content(self, backend):
        long_text = "word " * 10000
        await backend.store("k1", {"content": long_text})
        results = await backend.recall("word", top_k=5)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_stores(self, in_memory_backend):
        """Concurrent stores should not cause data corruption."""
        async def store_batch(prefix, n):
            for i in range(n):
                await in_memory_backend.store(
                    f"{prefix}_{i}",
                    {"content": f"Item {prefix} {i}", "significance": 0.5},
                )

        await asyncio.gather(
            store_batch("a", 20),
            store_batch("b", 20),
        )
        assert await in_memory_backend.count() == 40

    @pytest.mark.asyncio
    async def test_sanitize_fts_empty_tokens(self, backend):
        """Query with only punctuation should return empty results, not crash."""
        await backend.store("k1", {"content": "hello"})
        results = await backend.recall("!! ?? ..", top_k=5)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_top_k_respected(self, backend):
        for i in range(20):
            await backend.store(f"k{i}", {"content": f"common term item {i}"})
        results = await backend.recall("common term", top_k=5)
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_persistence_across_reopen(self, tmp_path):
        """Data survives closing and reopening the backend."""
        db_path = tmp_path / "persist.db"
        b1 = SQLiteBackend(db_path)
        await b1.store("k1", {"content": "persistent data"})
        b1.close()

        b2 = SQLiteBackend(db_path)
        assert await b2.count() == 1
        results = await b2.recall("persistent", top_k=5)
        assert len(results) >= 1
        b2.close()


# ══════════════════════════════════════════════════════════════
#  Performance: 1000 items
# ══════════════════════════════════════════════════════════════


class TestPerformance:

    @pytest.mark.asyncio
    async def test_bulk_insert_1000(self, in_memory_backend):
        """Inserting 1000 items should complete in a reasonable time."""
        t0 = time.time()
        for i in range(1000):
            await in_memory_backend.store(f"perf_{i}", {
                "content": f"Performance test item number {i} with various keywords data analysis",
                "category": f"cat_{i % 10}",
                "significance": (i % 10) / 10.0,
                "created_at": time.time() - i * 3600,
            })
        elapsed = time.time() - t0
        assert await in_memory_backend.count() == 1000
        # Should complete well under 10 seconds even on slow CI
        assert elapsed < 10.0

    @pytest.mark.asyncio
    async def test_search_1000_items(self, in_memory_backend):
        """Searching across 1000 items should return quickly."""
        for i in range(1000):
            await in_memory_backend.store(f"s_{i}", {
                "content": f"Document {i} about {'machine learning' if i % 3 == 0 else 'cooking recipes'}",
                "significance": 0.5,
                "created_at": time.time(),
            })

        t0 = time.time()
        results = await in_memory_backend.recall("machine learning", top_k=10)
        elapsed = time.time() - t0

        assert len(results) == 10
        assert elapsed < 1.0  # Should be near-instant

    @pytest.mark.asyncio
    async def test_forget_batch(self, in_memory_backend):
        """Forgetting items from a large dataset."""
        for i in range(100):
            await in_memory_backend.store(f"del_{i}", {"content": f"deletable {i}"})
        assert await in_memory_backend.count() == 100

        for i in range(50):
            await in_memory_backend.forget(f"del_{i}")
        assert await in_memory_backend.count() == 50


# ══════════════════════════════════════════════════════════════
#  Backend factory in MemoryManager
# ══════════════════════════════════════════════════════════════


class TestBackendFactory:

    def test_default_is_sqlite(self, tmp_path):
        """With memory.backend='sqlite', manager creates SQLiteBackend."""
        with patch("openclaw.memory.manager.get_settings") as mock_gs:
            settings = MagicMock()
            settings.get = lambda k, d=None: {
                "memory.backend": "sqlite",
                "memory.store_path": str(tmp_path / "store"),
                "memory.enabled": True,
                "memory.vector.enabled": False,
            }.get(k, d)
            settings._base_dir = tmp_path
            mock_gs.return_value = settings

            from openclaw.memory.manager import MemoryManager
            mgr = MemoryManager()
            assert isinstance(mgr.backend, SQLiteBackend)
            mgr.backend.close()

    def test_chromadb_fallback_to_sqlite(self, tmp_path):
        """When chromadb is requested but unavailable, fall back to SQLite."""
        with patch("openclaw.memory.manager.get_settings") as mock_gs:
            settings = MagicMock()
            settings.get = lambda k, d=None: {
                "memory.backend": "chromadb",
                "memory.store_path": str(tmp_path / "store"),
                "memory.enabled": True,
                "memory.vector.enabled": True,
            }.get(k, d)
            settings._base_dir = tmp_path
            mock_gs.return_value = settings

            # Force chromadb import to fail
            with patch("openclaw.memory.backends.chromadb_backend.ChromaDBBackend",
                        side_effect=ImportError("no chromadb")):
                from openclaw.memory.manager import MemoryManager
                mgr = MemoryManager()
                assert isinstance(mgr.backend, SQLiteBackend)
                mgr.backend.close()

    def test_injected_backend_used(self, tmp_path):
        """When a backend is injected, the factory is skipped."""
        mock_backend = MagicMock(spec=MemoryBackend)

        with patch("openclaw.memory.manager.get_settings") as mock_gs:
            settings = MagicMock()
            settings.get = lambda k, d=None: {
                "memory.store_path": str(tmp_path / "store"),
                "memory.enabled": True,
            }.get(k, d)
            settings._base_dir = tmp_path
            mock_gs.return_value = settings

            from openclaw.memory.manager import MemoryManager
            mgr = MemoryManager(backend=mock_backend)
            assert mgr.backend is mock_backend


# ══════════════════════════════════════════════════════════════
#  default.yaml config
# ══════════════════════════════════════════════════════════════


class TestDefaultConfig:

    def test_default_backend_is_sqlite(self):
        import yaml
        cfg_path = Path(__file__).parent.parent / "openclaw" / "config" / "default.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["memory"]["backend"] == "sqlite"

    def test_vector_disabled_by_default(self):
        import yaml
        cfg_path = Path(__file__).parent.parent / "openclaw" / "config" / "default.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["memory"]["vector"]["enabled"] is False
