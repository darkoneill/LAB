"""
Skill Loader - Discovery and loading of skills.
"""

import importlib
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Optional

from openclaw.config.settings import get_settings
from .base import BaseSkill

logger = logging.getLogger("openclaw.skills.loader")


class SkillLoader:
    """Discovers and loads skills from configured paths."""

    def __init__(self):
        self.settings = get_settings()
        self.skills: dict[str, BaseSkill] = {}

    def discover_and_load(self):
        """Discover and load all available skills."""
        base = self.settings._base_dir or Path(__file__).parent.parent

        # Load builtin skills
        builtin_path = base / self.settings.get("skills.builtin_path", "skills/builtin")
        if builtin_path.exists():
            self._load_from_directory(builtin_path)

        # Load custom skills
        custom_path = base / self.settings.get("skills.custom_path", "skills/custom")
        if custom_path.exists():
            self._load_from_directory(custom_path)

        logger.info(f"Loaded {len(self.skills)} skills")

    def _load_from_directory(self, directory: Path):
        """Load all skills from a directory."""
        for skill_dir in directory.iterdir():
            if skill_dir.is_dir() and not skill_dir.name.startswith("_"):
                try:
                    self._load_skill(skill_dir)
                except Exception as e:
                    logger.warning(f"Failed to load skill from {skill_dir}: {e}")

    def _load_skill(self, skill_dir: Path):
        """Load a single skill from its directory."""
        skill_py = skill_dir / "skill.py"
        if not skill_py.exists():
            return

        # Dynamic import
        module_name = f"openclaw.skills.loaded.{skill_dir.name}"
        spec = importlib.util.spec_from_file_location(module_name, skill_py)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find BaseSkill subclasses
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseSkill)
                    and attr is not BaseSkill
                ):
                    skill = attr(skill_path=skill_dir)
                    self.skills[skill.name] = skill
                    logger.info(f"Loaded skill: {skill.name}")

    def get_skill(self, name: str) -> Optional[BaseSkill]:
        return self.skills.get(name)

    def list_skills(self) -> list[dict]:
        return [s.get_info() for s in self.skills.values()]
