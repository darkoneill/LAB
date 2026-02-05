"""
Resource Layer - Raw data storage.
The foundation layer that stores all original data.
Resources are NEVER deleted (MemU principle).
"""

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openclaw.memory.resources")


class ResourceLayer:
    """
    Stores raw resources: conversations, files, knowledge, etc.
    Each resource is a JSON file for human readability (MemU philosophy).
    """

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = {}

    async def load(self):
        """Load resource index from disk."""
        self._index.clear()
        for f in sorted(self.store_path.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                rid = data.get("id", f.stem)
                self._index[rid] = {
                    "id": rid,
                    "type": data.get("type", "unknown"),
                    "timestamp": data.get("timestamp", 0),
                    "path": str(f),
                    "size": f.stat().st_size,
                }
            except Exception as e:
                logger.warning(f"Failed to load resource {f}: {e}")

        logger.info(f"Loaded {len(self._index)} resources")

    async def store(self, data: dict) -> str:
        """Store a new resource. Returns the resource ID."""
        rid = f"res_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        data["id"] = rid
        if "timestamp" not in data:
            data["timestamp"] = time.time()

        filepath = self.store_path / f"{rid}.json"
        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

        self._index[rid] = {
            "id": rid,
            "type": data.get("type", "unknown"),
            "timestamp": data["timestamp"],
            "path": str(filepath),
            "size": filepath.stat().st_size,
        }

        return rid

    async def get(self, resource_id: str) -> Optional[dict]:
        """Retrieve a resource by ID."""
        if resource_id not in self._index:
            return None
        filepath = Path(self._index[resource_id]["path"])
        if filepath.exists():
            return json.loads(filepath.read_text(encoding="utf-8"))
        return None

    async def search(self, query: str, resource_type: str = None, limit: int = 50) -> list[dict]:
        """Simple text search across resources."""
        results = []
        query_lower = query.lower()

        for rid, meta in self._index.items():
            if resource_type and meta["type"] != resource_type:
                continue
            filepath = Path(meta["path"])
            if filepath.exists():
                content = filepath.read_text(encoding="utf-8").lower()
                if query_lower in content:
                    results.append(meta)
                    if len(results) >= limit:
                        break
        return results

    @property
    def count(self) -> int:
        return len(self._index)
