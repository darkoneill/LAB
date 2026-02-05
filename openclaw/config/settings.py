"""
Configuration management with YAML loading, validation, and runtime updates.
"""

import os
import yaml
import copy
from pathlib import Path
from typing import Any, Optional


class Settings:
    """Centralized configuration manager with hot-reload support."""

    _instance: Optional["Settings"] = None
    _config: dict = {}
    _config_path: Optional[Path] = None
    _user_config_path: Optional[Path] = None
    _base_dir: Optional[Path] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, base_dir: Optional[str] = None):
        """Load configuration from default + user overrides."""
        instance = cls()
        if base_dir:
            instance._base_dir = Path(base_dir)
        else:
            instance._base_dir = Path(__file__).parent.parent

        default_path = instance._base_dir / "config" / "default.yaml"
        user_path = instance._base_dir / "config" / "user.yaml"
        instance._config_path = default_path
        instance._user_config_path = user_path

        # Load defaults
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                instance._config = yaml.safe_load(f) or {}

        # Overlay user config
        if user_path.exists():
            with open(user_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            instance._config = cls._deep_merge(instance._config, user_cfg)

        # Environment variable overrides (OPENCLAW_SECTION__KEY format)
        instance._apply_env_overrides()

        return instance

    def get(self, dotpath: str, default: Any = None) -> Any:
        """Get a config value using dot notation: 'gateway.port'."""
        keys = dotpath.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

    def set(self, dotpath: str, value: Any, persist: bool = False):
        """Set a config value at runtime. Optionally persist to user.yaml."""
        keys = dotpath.split(".")
        cfg = self._config
        for k in keys[:-1]:
            if k not in cfg or not isinstance(cfg[k], dict):
                cfg[k] = {}
            cfg = cfg[k]
        cfg[keys[-1]] = value

        if persist and self._user_config_path:
            self._save_user_config(dotpath, value)

    def get_section(self, section: str) -> dict:
        """Get an entire config section."""
        return copy.deepcopy(self._config.get(section, {}))

    def all(self) -> dict:
        """Return full configuration (deep copy)."""
        return copy.deepcopy(self._config)

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a path relative to the base directory."""
        p = Path(relative_path)
        if p.is_absolute():
            return p
        return self._base_dir / p

    def _save_user_config(self, dotpath: str, value: Any):
        """Persist a setting to user.yaml."""
        user_cfg = {}
        if self._user_config_path.exists():
            with open(self._user_config_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}

        keys = dotpath.split(".")
        cfg = user_cfg
        for k in keys[:-1]:
            if k not in cfg or not isinstance(cfg[k], dict):
                cfg[k] = {}
            cfg = cfg[k]
        cfg[keys[-1]] = value

        self._user_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._user_config_path, "w", encoding="utf-8") as f:
            yaml.dump(user_cfg, f, default_flow_style=False, allow_unicode=True)

    def _apply_env_overrides(self):
        """Apply OPENCLAW_* environment variables as config overrides."""
        prefix = "OPENCLAW_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_path = key[len(prefix):].lower().replace("__", ".")
                # Auto-cast
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                else:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                self.set(config_path, value)

    @staticmethod
    def _deep_merge(base: dict, overlay: dict) -> dict:
        """Deep merge overlay into base."""
        result = copy.deepcopy(base)
        for key, value in overlay.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Settings._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result


# Global convenience function
def get_settings() -> Settings:
    """Get the global Settings instance."""
    if Settings._instance is None:
        Settings.initialize()
    return Settings._instance
