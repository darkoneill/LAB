"""
Category Layer - Aggregated, human-readable memory files.
Memory items are organized into coherent documents (MemU's key innovation).
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openclaw.memory.categories")


# Default categories with descriptions
DEFAULT_CATEGORIES = {
    "user_profile": {
        "name": "Profil Utilisateur",
        "description": "Identity, preferences, personal details",
        "icon": "person",
    },
    "user_preferences": {
        "name": "Preferences",
        "description": "Likes, dislikes, styles, habits",
        "icon": "heart",
    },
    "knowledge": {
        "name": "Connaissances",
        "description": "Facts, concepts, learned information",
        "icon": "book",
    },
    "projects": {
        "name": "Projets",
        "description": "Active projects, goals, roadmaps",
        "icon": "folder",
    },
    "conversations": {
        "name": "Conversations",
        "description": "Key conversation topics and decisions",
        "icon": "chat",
    },
    "technical": {
        "name": "Technique",
        "description": "Code, architecture, configurations",
        "icon": "code",
    },
    "decisions": {
        "name": "Decisions",
        "description": "Important decisions and their rationale",
        "icon": "check",
    },
    "general": {
        "name": "General",
        "description": "Uncategorized memories",
        "icon": "star",
    },
}


class CategoryLayer:
    """
    Organizes memory items into structured, human-readable documents.
    Each category is a markdown file that can be inspected and edited.
    """

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._categories: dict[str, dict] = {}
        self._meta_path = self.store_path / "_meta.json"

    async def load(self):
        """Load categories from disk."""
        # Load metadata
        if self._meta_path.exists():
            try:
                self._categories = json.loads(self._meta_path.read_text(encoding="utf-8"))
            except Exception as e:
                logger.warning(f"Failed to load category metadata: {e}")

        # Ensure default categories exist
        for cat_id, cat_info in DEFAULT_CATEGORIES.items():
            if cat_id not in self._categories:
                self._categories[cat_id] = {
                    **cat_info,
                    "id": cat_id,
                    "item_count": 0,
                    "last_updated": 0,
                    "created_at": time.time(),
                }

        await self._persist_meta()
        logger.info(f"Loaded {len(self._categories)} categories")

    async def organize(self, items: list[dict]):
        """Organize new items into appropriate categories."""
        for item in items:
            category = item.get("category", "general")
            if category not in self._categories:
                # Create new category dynamically
                self._categories[category] = {
                    "id": category,
                    "name": category.replace("_", " ").title(),
                    "description": f"Auto-created for: {category}",
                    "icon": "folder",
                    "item_count": 0,
                    "last_updated": 0,
                    "created_at": time.time(),
                }

            # Update category file
            cat_file = self.store_path / f"{category}.md"
            content = item.get("content", "")
            if content:
                # Append to the category markdown file
                existing = ""
                if cat_file.exists():
                    existing = cat_file.read_text(encoding="utf-8")
                else:
                    cat_name = self._categories[category].get("name", category)
                    existing = f"# {cat_name}\n\n"

                timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(item.get("created_at", time.time())))
                significance = item.get("significance", 0.5)
                entry = f"- [{timestamp}] (sig:{significance:.1f}) {content}\n"

                if content not in existing:  # Avoid duplicates
                    cat_file.write_text(existing + entry, encoding="utf-8")
                    self._categories[category]["item_count"] = self._categories[category].get("item_count", 0) + 1
                    self._categories[category]["last_updated"] = time.time()

        await self._persist_meta()

    async def evolve(self):
        """
        Self-evolution: review categories and generate insights.
        This is the 'reflection' step from MemU.
        """
        logger.info("Running memory evolution cycle...")

        for cat_id, cat_meta in self._categories.items():
            cat_file = self.store_path / f"{cat_id}.md"
            if not cat_file.exists():
                continue

            content = cat_file.read_text(encoding="utf-8")
            lines = content.strip().split("\n")

            # If category is getting too large, consolidate
            if len(lines) > 100:
                logger.info(f"Consolidating category: {cat_id} ({len(lines)} lines)")
                await self._consolidate_category(cat_id, lines)

    async def _consolidate_category(self, cat_id: str, lines: list[str]):
        """Consolidate a large category file by grouping related items."""
        cat_file = self.store_path / f"{cat_id}.md"
        header = lines[0] if lines and lines[0].startswith("#") else f"# {cat_id}"

        # Keep the most recent and significant entries
        entries = [l for l in lines[1:] if l.strip().startswith("- ")]

        # Sort by significance (extracted from entry)
        def get_sig(entry):
            try:
                sig_start = entry.find("(sig:")
                if sig_start >= 0:
                    sig_end = entry.find(")", sig_start)
                    return float(entry[sig_start + 5:sig_end])
            except (ValueError, IndexError):
                pass
            return 0.5

        entries.sort(key=get_sig, reverse=True)

        # Keep top entries, summarize the rest
        keep = entries[:50]
        archived = entries[50:]

        new_content = header + "\n\n"
        new_content += "\n".join(keep) + "\n"

        if archived:
            new_content += f"\n---\n_[{len(archived)} older entries archived]_\n"

            # Save archived entries separately
            archive_file = self.store_path / f"{cat_id}_archive.md"
            archive_content = f"# {cat_id} - Archive\n\n" + "\n".join(archived) + "\n"
            archive_file.write_text(archive_content, encoding="utf-8")

        cat_file.write_text(new_content, encoding="utf-8")

    def get_category_content(self, category: str) -> str:
        """Read a category file."""
        cat_file = self.store_path / f"{category}.md"
        if cat_file.exists():
            return cat_file.read_text(encoding="utf-8")
        return ""

    def list_all(self) -> list[dict]:
        """List all categories with metadata."""
        result = []
        for cat_id, meta in self._categories.items():
            result.append({
                "id": cat_id,
                "name": meta.get("name", cat_id),
                "description": meta.get("description", ""),
                "item_count": meta.get("item_count", 0),
                "last_updated": meta.get("last_updated", 0),
            })
        return sorted(result, key=lambda x: x.get("item_count", 0), reverse=True)

    def search_categories(self, query: str, limit: int = 20) -> list[dict]:
        """Search across all category files."""
        results = []
        query_lower = query.lower()

        for cat_id in self._categories:
            content = self.get_category_content(cat_id)
            if query_lower in content.lower():
                # Extract matching lines
                for line in content.split("\n"):
                    if query_lower in line.lower() and line.strip().startswith("- "):
                        results.append({
                            "category": cat_id,
                            "content": line.strip("- ").strip(),
                            "source": "category",
                        })
                        if len(results) >= limit:
                            return results
        return results

    async def _persist_meta(self):
        """Save category metadata."""
        self._meta_path.write_text(
            json.dumps(self._categories, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    @property
    def count(self) -> int:
        return len(self._categories)
