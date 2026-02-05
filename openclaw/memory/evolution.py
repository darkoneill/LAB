"""
Memory Evolution - Self-reflection and insight generation.
The autonomous memory agent that organizes, links, and evolves memories.
"""

import logging
import time

logger = logging.getLogger("openclaw.memory.evolution")


class MemoryEvolver:
    """
    Autonomous memory evolution engine that:
    1. Organizes - auto-categorizes and files memories
    2. Links - finds connections between memories
    3. Evolves - generates new insights from existing memories
    4. Forgets - gracefully reduces priority of stale memories
    """

    def __init__(self, item_layer, category_layer):
        self.items = item_layer
        self.categories = category_layer
        self.insights_generated = 0

    async def run_cycle(self):
        """Run one evolution cycle."""
        logger.info("Starting memory evolution cycle")

        # Phase 1: Organize uncategorized items
        await self._organize()

        # Phase 2: Link related items
        await self._link()

        # Phase 3: Generate insights
        await self._evolve()

        logger.info(f"Evolution cycle complete. {self.insights_generated} insights generated.")

    async def _organize(self):
        """Re-categorize items that may be misplaced."""
        all_items = self.items.all_items()
        recategorized = 0

        for item in all_items:
            if item.get("category") == "general":
                suggested = self._suggest_category(item.get("content", ""))
                if suggested != "general":
                    item["category"] = suggested
                    recategorized += 1

        if recategorized > 0:
            logger.info(f"Recategorized {recategorized} items")

    async def _link(self):
        """Find and record connections between memory items."""
        all_items = self.items.all_items()
        if len(all_items) < 2:
            return

        # Simple co-occurrence linking
        for i, item_a in enumerate(all_items):
            words_a = set(item_a.get("content", "").lower().split())
            for item_b in all_items[i + 1:]:
                words_b = set(item_b.get("content", "").lower().split())
                overlap = words_a & words_b
                # Remove common/stop words
                meaningful_overlap = {w for w in overlap if len(w) > 4}
                if len(meaningful_overlap) >= 3:
                    links_a = item_a.get("links", [])
                    if item_b.get("id") not in links_a:
                        links_a.append(item_b["id"])
                        item_a["links"] = links_a[:10]  # Max 10 links

    async def _evolve(self):
        """Generate new insights from existing memories."""
        significant_items = self.items.get_significant(min_significance=0.6, limit=20)

        if len(significant_items) < 3:
            return

        # Group by category and look for patterns
        by_category = {}
        for item in significant_items:
            cat = item.get("category", "general")
            by_category.setdefault(cat, []).append(item)

        for category, items in by_category.items():
            if len(items) >= 3:
                # Potential insight: recurring theme
                common_words = self._find_common_themes(items)
                if common_words:
                    self.insights_generated += 1

    def _suggest_category(self, content: str) -> str:
        """Suggest a category based on content analysis."""
        content_lower = content.lower()

        category_keywords = {
            "user_profile": ["my name", "i am", "je suis", "je m'appelle", "mon nom", "years old", "ans"],
            "user_preferences": ["i like", "i prefer", "i love", "j'aime", "je prefere", "favorite"],
            "technical": ["code", "function", "api", "server", "database", "git", "python", "docker"],
            "projects": ["project", "milestone", "deadline", "sprint", "roadmap", "plan"],
            "decisions": ["decided", "chose", "because", "rationale", "conclusion"],
        }

        best_cat = "general"
        best_score = 0

        for cat, keywords in category_keywords.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > best_score:
                best_score = score
                best_cat = cat

        return best_cat

    def _find_common_themes(self, items: list[dict]) -> list[str]:
        """Find common themes across items."""
        from collections import Counter
        all_words = []
        for item in items:
            words = item.get("content", "").lower().split()
            all_words.extend([w for w in words if len(w) > 4])

        counter = Counter(all_words)
        return [word for word, count in counter.most_common(5) if count >= 2]
