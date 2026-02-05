"""
Web UI - FastAPI-based web interface for OpenClaw.
"""

import logging
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger("openclaw.ui.web")

WEB_DIR = Path(__file__).parent


def setup_web_ui(app):
    """Mount the web UI on the FastAPI app."""
    static_dir = WEB_DIR / "static"
    templates_dir = WEB_DIR / "templates"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    templates = Jinja2Templates(directory=str(templates_dir))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})
