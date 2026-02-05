"""OpenClaw - Package Setup"""

from setuptools import setup, find_packages
from openclaw import __version__

setup(
    name="openclaw",
    version=__version__,
    description="Autonomous AI Assistant - OpenClaw + MemU + AgentZero",
    author="OpenClaw Contributors",
    python_requires=">=3.10",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "openclaw": [
            "config/*.yaml",
            "agent/prompts/*.md",
            "skills/builtin/*/SKILL.md",
            "ui/web/templates/*.html",
            "ui/web/static/css/*.css",
            "ui/web/static/js/*.js",
        ],
    },
    entry_points={
        "console_scripts": [
            "openclaw=openclaw.main:main",
        ],
    },
    install_requires=[
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.32.0",
        "websockets>=13.0",
        "httpx>=0.27.0",
        "pydantic>=2.9.0",
        "pyyaml>=6.0.2",
        "rich>=13.9.0",
        "prompt_toolkit>=3.0.48",
        "jinja2>=3.1.4",
        "aiofiles>=24.1.0",
    ],
    extras_require={
        "anthropic": ["anthropic>=0.39.0"],
        "openai": ["openai>=1.55.0"],
        "all": ["anthropic>=0.39.0", "openai>=1.55.0"],
    },
)
