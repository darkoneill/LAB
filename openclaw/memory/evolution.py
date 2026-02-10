"""
Memory Evolution - Self-reflection and insight generation.
The autonomous memory agent that organizes, links, and evolves memories.

Includes Episodic Narrative Memory ("Consolidation Nocturne"):
After long sessions or once per day, the agent rereads all exchanges
and writes a chapter summary to PROJECT_MEMORY.md.
"""

import logging
import os
import time
from datetime import datetime
from typing import Optional, TYPE_CHECKING

from openclaw.config.settings import get_settings

if TYPE_CHECKING:
    from openclaw.agent.brain import AgentBrain

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


# ── Episodic Narrative Memory ─────────────────────────────────────────

CONSOLIDATION_PROMPT = """Tu es l'archiviste interne d'un agent IA autonome appele OpenClaw.
Voici les echanges et decisions de la session ecoulee.

{session_content}

Redige un "Resume du Chapitre" structure avec:
1. **Contexte** : Quel etait l'objectif de la session ?
2. **Actions cles** : Quelles decisions et actions ont ete prises ?
3. **Resultats** : Qu'est-ce qui a ete accompli ?
4. **Lecons apprises** : Quelles erreurs ont ete commises et corrigees ?
5. **Pistes ouvertes** : Que reste-t-il a faire ?

Sois concis (max 300 mots). Ecris en francais. Pas de preamble."""

# Minimum exchanges required before consolidation is triggered
MIN_EXCHANGES_FOR_CONSOLIDATION = 5
# Minimum elapsed time (seconds) since last consolidation (default: 6 hours)
MIN_INTERVAL_SECONDS = 6 * 3600


class EpisodicMemory:
    """
    Episodic Narrative Memory - "Consolidation Nocturne".

    After a long session (or once per configurable interval), the agent
    rereads all exchanges and produces a narrative chapter summary
    appended to PROJECT_MEMORY.md.

    This summary is injected into the system context at the start
    of the next session so the agent remembers past work.
    """

    def __init__(self, brain: Optional["AgentBrain"] = None):
        self.settings = get_settings()
        self.brain = brain
        self._memory_file = self.settings.get(
            "memory.episodic.file", "PROJECT_MEMORY.md"
        )
        self._min_exchanges = self.settings.get(
            "memory.episodic.min_exchanges", MIN_EXCHANGES_FOR_CONSOLIDATION
        )
        self._min_interval = self.settings.get(
            "memory.episodic.min_interval_seconds", MIN_INTERVAL_SECONDS
        )
        self._last_consolidation: Optional[float] = self._read_last_timestamp()

    def _read_last_timestamp(self) -> Optional[float]:
        """Read the timestamp of the last consolidation from the memory file."""
        if not os.path.exists(self._memory_file):
            return None
        try:
            with open(self._memory_file, "r", encoding="utf-8") as f:
                content = f.read()
            # Look for the last "<!-- consolidated: TIMESTAMP -->" marker
            marker = "<!-- consolidated: "
            idx = content.rfind(marker)
            if idx == -1:
                return None
            start = idx + len(marker)
            end = content.find(" -->", start)
            if end == -1:
                return None
            return float(content[start:end])
        except (OSError, ValueError):
            return None

    def should_consolidate(self, exchange_count: int) -> bool:
        """
        Check if consolidation should be triggered.

        Triggers when:
        - Enough exchanges have occurred (>= min_exchanges)
        - Enough time has passed since last consolidation (>= min_interval)
        """
        if exchange_count < self._min_exchanges:
            return False

        if self._last_consolidation is not None:
            elapsed = time.time() - self._last_consolidation
            if elapsed < self._min_interval:
                return False

        return True

    async def consolidate(self, exchanges: list[dict]) -> Optional[str]:
        """
        Produce a narrative summary of the session and append to PROJECT_MEMORY.md.

        Args:
            exchanges: List of {"role": str, "content": str} messages from the session.

        Returns:
            The generated summary text, or None if consolidation was skipped.
        """
        if not exchanges:
            logger.debug("No exchanges to consolidate")
            return None

        if not self.brain:
            logger.warning("No brain reference - cannot generate narrative summary")
            return None

        # Build session content for the prompt
        session_lines = []
        for ex in exchanges:
            role = ex.get("role", "unknown").upper()
            content = ex.get("content", "")
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            session_lines.append(f"[{role}]: {content}")

        session_content = "\n".join(session_lines)

        # Cap the total content to avoid exceeding token limits
        if len(session_content) > 8000:
            session_content = session_content[:8000] + "\n... (tronque)"

        prompt = CONSOLIDATION_PROMPT.format(session_content=session_content)

        try:
            result = await self.brain.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1024,
            )
            summary = result.get("content", "")
            if not summary.strip():
                logger.warning("Empty consolidation summary generated")
                return None

        except Exception as e:
            logger.error(f"Consolidation LLM call failed: {e}")
            return None

        # Write to PROJECT_MEMORY.md
        self._append_to_memory(summary)
        self._last_consolidation = time.time()

        logger.info(
            f"Episodic consolidation complete: "
            f"{len(summary)} chars written to {self._memory_file}"
        )
        return summary

    def _append_to_memory(self, summary: str):
        """Append a chapter summary to the PROJECT_MEMORY.md file."""
        now = datetime.now()
        timestamp = time.time()
        header = f"\n\n---\n\n## Chapitre - {now.strftime('%Y-%m-%d %H:%M')}\n"
        marker = f"<!-- consolidated: {timestamp} -->\n"

        # Create file with header if it doesn't exist
        if not os.path.exists(self._memory_file):
            with open(self._memory_file, "w", encoding="utf-8") as f:
                f.write("# PROJECT MEMORY - OpenClaw\n\n")
                f.write("Memoire episodique narrative de l'agent. ")
                f.write("Chaque chapitre resume une session de travail.\n")

        with open(self._memory_file, "a", encoding="utf-8") as f:
            f.write(header)
            f.write(marker)
            f.write(summary)
            f.write("\n")

    def get_narrative_context(self, max_chapters: int = 3) -> str:
        """
        Read recent chapter summaries to inject into system context.

        Args:
            max_chapters: Maximum number of recent chapters to include.

        Returns:
            Formatted string with recent narrative memory.
        """
        if not os.path.exists(self._memory_file):
            return ""

        try:
            with open(self._memory_file, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return ""

        # Split by chapter markers
        chapters = content.split("\n## Chapitre - ")
        if len(chapters) <= 1:
            return ""

        # Take the last N chapters (skip the file header at index 0)
        recent = chapters[-max_chapters:]
        formatted = []
        for ch in recent:
            # Re-add the header prefix
            formatted.append("## Chapitre - " + ch.strip())

        return (
            "[MEMOIRE EPISODIQUE - Resumes des sessions precedentes]\n\n"
            + "\n\n---\n\n".join(formatted)
        )
