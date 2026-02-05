"""
Base Skill - Foundation for all skills.
Skills are modular capability extensions, each with a SKILL.md config.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger("openclaw.skills")


class BaseSkill(ABC):
    """
    Base class for all OpenClaw skills.

    Each skill directory contains:
    - SKILL.md: Metadata and configuration (YAML frontmatter + description)
    - skill.py: Implementation (this class)
    - Optional: additional scripts, data, templates
    """

    name: str = "unnamed_skill"
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: list[str] = []
    dependencies: list[str] = []

    def __init__(self, skill_path: Path = None):
        self.skill_path = skill_path
        self.enabled = True
        self._load_metadata()

    def _load_metadata(self):
        """Load SKILL.md metadata if available."""
        if not self.skill_path:
            return

        skill_md = self.skill_path / "SKILL.md"
        if skill_md.exists():
            content = skill_md.read_text(encoding="utf-8")
            self._parse_frontmatter(content)

    def _parse_frontmatter(self, content: str):
        """Parse YAML frontmatter from SKILL.md."""
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    import yaml
                    meta = yaml.safe_load(parts[1])
                    if isinstance(meta, dict):
                        self.name = meta.get("name", self.name)
                        self.description = meta.get("description", self.description)
                        self.version = meta.get("version", self.version)
                        self.author = meta.get("author", self.author)
                        self.tags = meta.get("tags", self.tags)
                        self.dependencies = meta.get("dependencies", self.dependencies)
                except Exception:
                    pass

    @abstractmethod
    async def execute(self, **kwargs) -> dict:
        """Execute the skill with given parameters."""
        pass

    def matches(self, intent: str) -> float:
        """
        Return a confidence score (0-1) of how well this skill matches the intent.
        Override for custom matching logic.
        """
        intent_lower = intent.lower()
        name_lower = self.name.lower()
        desc_lower = self.description.lower()

        # Direct name match
        if name_lower in intent_lower:
            return 0.9

        # Tag match
        for tag in self.tags:
            if tag.lower() in intent_lower:
                return 0.7

        # Description keyword overlap
        intent_words = set(intent_lower.split())
        desc_words = set(desc_lower.split())
        overlap = intent_words & desc_words
        if overlap:
            return min(0.6, len(overlap) / max(len(intent_words), 1))

        return 0.0

    def get_info(self) -> dict:
        """Return skill info dict."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "enabled": self.enabled,
        }
