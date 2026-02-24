"""
SQLite FTS5 memory backend — lightweight alternative to ChromaDB.
BM25 keyword search + significance/recency scoring. No embeddings needed.
RAM usage: ~50MB vs ~500MB for ChromaDB+sentence-transformers.
"""

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path

from openclaw.memory.backends.base import MemoryBackend

logger = logging.getLogger("openclaw.memory.backends.sqlite")

# Weight split: BM25 keyword relevance vs significance/recency boost
BM25_WEIGHT = 0.6
SIGNIFICANCE_WEIGHT = 0.4


class SQLiteBackend(MemoryBackend):
    """MemoryBackend backed by SQLite with FTS5 full-text search."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()

    # ── MemoryBackend interface ────────────────────────────

    async def store(self, key: str, data: dict) -> None:
        conn = self._get_conn()
        content = data.get("content", "")
        category = data.get("category", "general")
        significance = data.get("significance", 0.5)
        access_count = data.get("access_count", 0)
        created_at = data.get("created_at", time.time())
        metadata = {k: v for k, v in data.items()
                    if k not in ("content", "category", "significance",
                                 "access_count", "created_at")}

        async with self._lock:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories
                    (id, content, category, significance, access_count,
                     created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (key, content, category, significance, access_count,
                 created_at, json.dumps(metadata, default=str)),
            )
            conn.commit()

    async def recall(self, query: str, top_k: int = 10) -> list[dict]:
        conn = self._get_conn()
        if not query or not query.strip():
            return []

        fts_query = self._sanitize_fts_query(query)

        async with self._lock:
            # FTS5 BM25 search
            if fts_query:
                rows = conn.execute(
                    """
                    SELECT
                        m.id, m.content, m.category, m.significance,
                        m.access_count, m.created_at, m.metadata_json,
                        bm25(memories_fts) AS bm25_score
                    FROM memories_fts AS fts
                    JOIN memories AS m ON m.id = fts.rowid_ref
                    WHERE memories_fts MATCH ?
                    ORDER BY bm25_score
                    LIMIT ?
                    """,
                    (fts_query, top_k * 3),  # over-fetch for re-ranking
                ).fetchall()
            else:
                rows = []

            # Fall back to LIKE search if FTS5 produced nothing
            if not rows:
                like_terms = [f"%{t}%" for t in query.lower().split()[:5]]
                clauses = " OR ".join(["LOWER(content) LIKE ?"] * len(like_terms))
                rows = conn.execute(
                    f"""
                    SELECT id, content, category, significance,
                           access_count, created_at, metadata_json,
                           0.0 AS bm25_score
                    FROM memories
                    WHERE {clauses}
                    ORDER BY significance DESC
                    LIMIT ?
                    """,
                    (*like_terms, top_k * 3),
                ).fetchall()

        # Re-rank with hybrid score
        now = time.time()
        scored = []
        for row in rows:
            (rid, content, category, significance,
             access_count, created_at, meta_json, bm25_raw) = row

            # BM25 returns negative values (lower = more relevant)
            # Normalise to [0, 1] range: score = 1 / (1 + abs(bm25))
            bm25_norm = 1.0 / (1.0 + abs(bm25_raw))

            # Recency boost: decays over 30 days
            age_days = max(0, (now - created_at) / 86400)
            recency = 1.0 / (1.0 + age_days / 30.0)

            sig_recency = (significance * 0.7 + recency * 0.3)
            hybrid_score = BM25_WEIGHT * bm25_norm + SIGNIFICANCE_WEIGHT * sig_recency

            meta = {}
            if meta_json:
                try:
                    meta = json.loads(meta_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            scored.append({
                "id": rid,
                "content": content,
                "category": category,
                "significance": significance,
                "access_count": access_count,
                "created_at": created_at,
                "score": round(hybrid_score, 4),
                "metadata": meta,
            })

        # Sort descending by hybrid score, return top_k
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    async def forget(self, key: str) -> None:
        conn = self._get_conn()
        async with self._lock:
            conn.execute("DELETE FROM memories WHERE id = ?", (key,))
            conn.commit()

    async def count(self) -> int:
        conn = self._get_conn()
        async with self._lock:
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    # ── Extra helpers (not part of ABC) ────────────────────

    async def get(self, key: str) -> dict | None:
        """Retrieve a single item by key and bump access_count."""
        conn = self._get_conn()
        async with self._lock:
            row = conn.execute(
                "SELECT id, content, category, significance, "
                "access_count, created_at, metadata_json "
                "FROM memories WHERE id = ?",
                (key,),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                (key,),
            )
            conn.commit()

        (rid, content, category, significance,
         access_count, created_at, meta_json) = row
        meta = {}
        if meta_json:
            try:
                meta = json.loads(meta_json)
            except (json.JSONDecodeError, TypeError):
                pass

        return {
            "id": rid,
            "content": content,
            "category": category,
            "significance": significance,
            "access_count": access_count + 1,
            "created_at": created_at,
            "metadata": meta,
        }

    async def update_significance(self, key: str, significance: float) -> None:
        """Update the significance score for a given item."""
        conn = self._get_conn()
        async with self._lock:
            conn.execute(
                "UPDATE memories SET significance = ? WHERE id = ?",
                (significance, key),
            )
            conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── Internal ───────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._create_schema()
        return self._conn

    def _create_schema(self) -> None:
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL DEFAULT '',
                category TEXT NOT NULL DEFAULT 'general',
                significance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                metadata_json TEXT DEFAULT '{}'
            )
        """)
        # FTS5 virtual table using content-sync (external content not needed;
        # we JOIN via a stored rowid_ref column for simplicity).
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
            USING fts5(
                content_col,
                category_col,
                rowid_ref UNINDEXED,
                tokenize='porter unicode61'
            )
        """)
        # Triggers to keep FTS in sync
        c.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories
            BEGIN
                INSERT INTO memories_fts(content_col, category_col, rowid_ref)
                VALUES (NEW.content, NEW.category, NEW.id);
            END
        """)
        c.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories
            BEGIN
                DELETE FROM memories_fts WHERE rowid_ref = OLD.id;
            END
        """)
        c.execute("""
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories
            BEGIN
                DELETE FROM memories_fts WHERE rowid_ref = OLD.id;
                INSERT INTO memories_fts(content_col, category_col, rowid_ref)
                VALUES (NEW.content, NEW.category, NEW.id);
            END
        """)
        c.commit()

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        """Turn a free-form query into a safe FTS5 MATCH expression.

        FTS5 special characters are stripped; each remaining token is
        joined with implicit AND so all terms must appear.
        """
        # Remove FTS5 operators and special characters
        cleaned = ""
        for ch in query:
            if ch.isalnum() or ch in (" ", "-", "'"):
                cleaned += ch
        tokens = [t.strip() for t in cleaned.split() if len(t.strip()) >= 2]
        if not tokens:
            return ""
        # Quote each token to avoid accidental operators
        return " ".join(f'"{t}"' for t in tokens)
