"""System Info Skill - System information and monitoring."""

import os
import platform
import subprocess

from openclaw.skills.base import BaseSkill


class SystemInfoSkill(BaseSkill):
    name = "system_info"
    description = "Get system information: OS, CPU, memory, disk, network"
    tags = ["system", "info", "os", "cpu", "memory", "disk", "hardware", "status"]

    async def execute(self, **kwargs) -> dict:
        section = kwargs.get("section", "all")

        info = {}

        if section in ("all", "os"):
            info["os"] = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            }

        if section in ("all", "cpu"):
            info["cpu"] = self._get_cpu_info()

        if section in ("all", "memory"):
            info["memory"] = self._get_memory_info()

        if section in ("all", "disk"):
            info["disk"] = self._get_disk_info()

        if section in ("all", "env"):
            info["environment"] = {
                "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
                "home": os.path.expanduser("~"),
                "cwd": os.getcwd(),
                "path_entries": len(os.environ.get("PATH", "").split(os.pathsep)),
            }

        return {"success": True, **info}

    def _get_cpu_info(self) -> dict:
        try:
            cpu_count = os.cpu_count() or 0
            load = os.getloadavg() if hasattr(os, "getloadavg") else (0, 0, 0)
            return {
                "count": cpu_count,
                "load_1m": round(load[0], 2),
                "load_5m": round(load[1], 2),
                "load_15m": round(load[2], 2),
            }
        except Exception:
            return {"count": os.cpu_count() or 0}

    def _get_memory_info(self) -> dict:
        try:
            result = subprocess.run(
                ["free", "-m"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) >= 2:
                    parts = lines[1].split()
                    return {
                        "total_mb": int(parts[1]),
                        "used_mb": int(parts[2]),
                        "free_mb": int(parts[3]),
                        "available_mb": int(parts[6]) if len(parts) > 6 else int(parts[3]),
                    }
        except Exception:
            pass
        return {"info": "Memory info unavailable"}

    def _get_disk_info(self) -> dict:
        try:
            stat = os.statvfs("/")
            total = (stat.f_blocks * stat.f_frsize) // (1024 ** 3)
            free = (stat.f_bavail * stat.f_frsize) // (1024 ** 3)
            used = total - free
            return {
                "total_gb": total,
                "used_gb": used,
                "free_gb": free,
                "usage_percent": round((used / total) * 100, 1) if total > 0 else 0,
            }
        except Exception:
            return {"info": "Disk info unavailable"}
