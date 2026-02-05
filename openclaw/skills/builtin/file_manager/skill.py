"""File Manager Skill - File system operations."""

import os
import shutil
from pathlib import Path

from openclaw.skills.base import BaseSkill


class FileManagerSkill(BaseSkill):
    name = "file_manager"
    description = "Read, write, search, and manage files and directories"
    tags = ["file", "read", "write", "directory", "search", "filesystem"]

    async def execute(self, **kwargs) -> dict:
        action = kwargs.get("action", "list")
        path = kwargs.get("path", ".")
        content = kwargs.get("content", "")

        try:
            if action == "read":
                return await self._read(path)
            elif action == "write":
                return await self._write(path, content)
            elif action == "list":
                return await self._list(path)
            elif action == "search":
                pattern = kwargs.get("pattern", "*")
                return await self._search(path, pattern)
            elif action == "info":
                return await self._info(path)
            elif action == "mkdir":
                return await self._mkdir(path)
            elif action == "delete":
                return await self._delete(path)
            elif action == "copy":
                dest = kwargs.get("destination", "")
                return await self._copy(path, dest)
            elif action == "move":
                dest = kwargs.get("destination", "")
                return await self._move(path, dest)
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _read(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {"success": False, "error": "File not found"}
        content = p.read_text(encoding="utf-8", errors="replace")
        return {"success": True, "content": content, "size": len(content)}

    async def _write(self, path: str, content: str) -> dict:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"success": True, "path": str(p), "size": len(content)}

    async def _list(self, path: str) -> dict:
        p = Path(path)
        if not p.exists() or not p.is_dir():
            return {"success": False, "error": "Directory not found"}
        entries = []
        for item in sorted(p.iterdir()):
            entries.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else 0,
            })
        return {"success": True, "entries": entries, "count": len(entries)}

    async def _search(self, path: str, pattern: str) -> dict:
        p = Path(path)
        matches = list(p.rglob(pattern))[:100]
        return {"success": True, "matches": [str(m) for m in matches], "count": len(matches)}

    async def _info(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {"success": False, "error": "Path not found"}
        stat = p.stat()
        return {
            "success": True,
            "path": str(p.resolve()),
            "type": "dir" if p.is_dir() else "file",
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "permissions": oct(stat.st_mode),
        }

    async def _mkdir(self, path: str) -> dict:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return {"success": True, "path": str(p)}

    async def _delete(self, path: str) -> dict:
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        return {"success": True, "deleted": str(p)}

    async def _copy(self, src: str, dest: str) -> dict:
        s, d = Path(src), Path(dest)
        if s.is_dir():
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
        return {"success": True, "source": str(s), "destination": str(d)}

    async def _move(self, src: str, dest: str) -> dict:
        shutil.move(src, dest)
        return {"success": True, "source": src, "destination": dest}
