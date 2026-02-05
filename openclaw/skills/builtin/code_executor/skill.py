"""Code Executor Skill - Run code in multiple languages."""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

from openclaw.skills.base import BaseSkill
from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.skills.code_executor")

LANGUAGE_CONFIG = {
    "python": {"cmd": "python3", "ext": ".py"},
    "javascript": {"cmd": "node", "ext": ".js"},
    "bash": {"cmd": "bash", "ext": ".sh"},
    "ruby": {"cmd": "ruby", "ext": ".rb"},
}


class CodeExecutorSkill(BaseSkill):
    name = "code_executor"
    description = "Execute code in Python, JavaScript, Bash and other languages"
    tags = ["code", "execute", "python", "javascript", "bash", "run", "script"]

    async def execute(self, **kwargs) -> dict:
        code = kwargs.get("code", "")
        language = kwargs.get("language", "python").lower()
        timeout = kwargs.get("timeout", None)

        if not code:
            return {"success": False, "error": "No code provided"}

        settings = get_settings()
        allowed_langs = settings.get("tools.code_executor.languages", list(LANGUAGE_CONFIG.keys()))
        if language not in allowed_langs:
            return {"success": False, "error": f"Language not allowed: {language}"}

        if language not in LANGUAGE_CONFIG:
            return {"success": False, "error": f"Unsupported language: {language}"}

        timeout = timeout or settings.get("tools.code_executor.timeout_seconds", 60)

        config = LANGUAGE_CONFIG[language]

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=config["ext"],
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(code)
                temp_path = f.name

            try:
                process = await asyncio.create_subprocess_exec(
                    config["cmd"],
                    temp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ},
                )

                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )

                return {
                    "success": process.returncode == 0,
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                    "return_code": process.returncode,
                    "language": language,
                }

            finally:
                os.unlink(temp_path)

        except asyncio.TimeoutError:
            return {"success": False, "error": f"Execution timed out ({timeout}s)"}
        except FileNotFoundError:
            return {"success": False, "error": f"Runtime not found: {config['cmd']}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
