"""
Skill Router - Matches user intent to appropriate skills.
"""

import logging
from typing import Optional

from .loader import SkillLoader
from .base import BaseSkill

logger = logging.getLogger("openclaw.skills.router")


class SkillRouter:
    """Routes user intents to the best matching skill."""

    def __init__(self, loader: SkillLoader = None):
        self.loader = loader or SkillLoader()
        self._threshold = 0.3  # Minimum confidence to route

    def initialize(self):
        """Load all skills."""
        self.loader.discover_and_load()

    def route(self, intent: str) -> Optional[BaseSkill]:
        """Find the best matching skill for a given intent."""
        best_skill = None
        best_score = 0

        for name, skill in self.loader.skills.items():
            if not skill.enabled:
                continue
            score = skill.matches(intent)
            if score > best_score and score >= self._threshold:
                best_score = score
                best_skill = skill

        if best_skill:
            logger.info(f"Routed intent to skill: {best_skill.name} (score: {best_score:.2f})")

        return best_skill

    async def execute(self, intent: str, **kwargs) -> Optional[dict]:
        """Find and execute the best matching skill."""
        skill = self.route(intent)
        if skill:
            return await skill.execute(**kwargs)
        return None

    def list_skills(self) -> list[dict]:
        return self.loader.list_skills()

    def get_skills_description(self) -> str:
        """Get a formatted description of all available skills."""
        skills = self.loader.list_skills()
        if not skills:
            return ""
        parts = []
        for s in skills:
            parts.append(f"- **{s['name']}**: {s['description']} (tags: {', '.join(s.get('tags', []))})")
        return "\n".join(parts)
