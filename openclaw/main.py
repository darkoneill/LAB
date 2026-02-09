"""
OpenClaw - Main Entry Point
Autonomous AI Assistant Framework
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure openclaw package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from openclaw import __version__, __codename__
from openclaw.config.settings import Settings, get_settings
from openclaw.gateway.server import GatewayServer
from openclaw.agent.brain import AgentBrain
from openclaw.agent.orchestrator import AgentOrchestrator
from openclaw.memory.manager import MemoryManager
from openclaw.skills.router import SkillRouter
from openclaw.skills.loader import SkillLoader
from openclaw.tools.executor import ToolExecutor
from openclaw.ui.terminal import TerminalUI
from openclaw.setup_wizard import SetupWizard


def setup_logging(level: str = "INFO"):
    """Configure logging."""
    settings = get_settings()
    log_level = getattr(logging, settings.get("logging.level", level).upper(), logging.INFO)
    log_format = settings.get("logging.format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Create log directory
    log_file = settings.get("logging.file", "logs/openclaw.log")
    log_path = settings.resolve_path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
    )
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def check_first_run(settings) -> bool:
    """Check if this is the first run (no user config)."""
    user_config = settings.resolve_path("config/user.yaml")
    return not user_config.exists()


async def initialize_components(settings):
    """Initialize all core components."""
    logger = logging.getLogger("openclaw.init")

    # Tool Executor
    tool_executor = ToolExecutor()
    logger.info("Tool executor initialized")

    # Skill System
    skill_loader = SkillLoader()
    skill_router = SkillRouter(loader=skill_loader)
    skill_router.initialize()
    logger.info("Skill system initialized")

    # Agent Brain
    agent_brain = AgentBrain(tool_executor=tool_executor, skill_router=skill_router)
    logger.info("Agent brain initialized")

    # Agent Orchestrator
    orchestrator = AgentOrchestrator(brain=agent_brain)
    logger.info("Agent orchestrator initialized")

    # Memory Manager
    memory_manager = None
    if settings.get("memory.enabled", True):
        memory_manager = MemoryManager()
        await memory_manager.initialize()
        logger.info("Memory system initialized")

    return {
        "tool_executor": tool_executor,
        "skill_router": skill_router,
        "agent_brain": agent_brain,
        "orchestrator": orchestrator,
        "memory_manager": memory_manager,
    }


async def run_terminal(components):
    """Run the terminal UI."""
    terminal = TerminalUI(
        agent_brain=components["agent_brain"],
        memory_manager=components["memory_manager"],
        orchestrator=components["orchestrator"],
    )
    await terminal.run()


async def run_gateway(components, settings):
    """Run the gateway server."""
    gateway = GatewayServer(
        agent_brain=components["agent_brain"],
        memory_manager=components["memory_manager"],
        skill_router=components["skill_router"],
    )

    # Mount Web UI
    if settings.get("ui.web.enabled", True):
        from openclaw.ui.web.app import setup_web_ui
        setup_web_ui(gateway.app)

    await gateway.start()


async def run_both(components, settings):
    """Run both terminal and gateway concurrently."""
    gateway = GatewayServer(
        agent_brain=components["agent_brain"],
        memory_manager=components["memory_manager"],
        skill_router=components["skill_router"],
    )

    # Mount Web UI
    if settings.get("ui.web.enabled", True):
        from openclaw.ui.web.app import setup_web_ui
        setup_web_ui(gateway.app)

    # Start gateway in background
    import uvicorn
    host = settings.get("gateway.host", "127.0.0.1")
    port = settings.get("gateway.port", 18789)

    config = uvicorn.Config(
        gateway.app,
        host=host,
        port=port,
        log_level="warning",
    )
    server = uvicorn.Server(config)

    # Run gateway and terminal concurrently
    gateway_task = asyncio.create_task(server.serve())

    terminal = TerminalUI(
        agent_brain=components["agent_brain"],
        memory_manager=components["memory_manager"],
        orchestrator=components["orchestrator"],
    )

    try:
        await terminal.run()
    finally:
        server.should_exit = True
        gateway_task.cancel()
        try:
            await gateway_task
        except asyncio.CancelledError:
            pass


async def main_async(args):
    """Async main entry point."""
    base_dir = str(Path(__file__).parent)
    settings = Settings.initialize(base_dir)
    setup_logging()

    logger = logging.getLogger("openclaw")
    logger.info(f"OpenClaw {__version__} ({__codename__}) starting...")

    # First run? Launch wizard
    if check_first_run(settings) and not args.no_wizard:
        wizard = SetupWizard()
        await wizard.run()
        # Reload settings after wizard
        settings = Settings.initialize(base_dir)

    # Initialize all components
    components = await initialize_components(settings)

    # Run the appropriate mode
    if args.mode == "terminal":
        await run_terminal(components)
    elif args.mode == "gateway":
        await run_gateway(components, settings)
    elif args.mode == "wizard":
        wizard = SetupWizard()
        await wizard.run()
    else:
        # Default: both terminal + gateway
        await run_both(components, settings)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="openclaw",
        description=f"OpenClaw {__version__} - Autonomous AI Assistant",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="both",
        choices=["terminal", "gateway", "both", "wizard"],
        help="Run mode: terminal (CLI only), gateway (API only), both (default), wizard (setup)",
    )
    parser.add_argument(
        "--no-wizard",
        action="store_true",
        help="Skip the setup wizard on first run",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Gateway host override",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Gateway port override",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"OpenClaw {__version__} ({__codename__})",
    )

    args = parser.parse_args()

    if args.debug:
        os.environ["OPENCLAW_APP__DEBUG"] = "true"
        os.environ["OPENCLAW_LOGGING__LEVEL"] = "DEBUG"

    if args.host:
        os.environ["OPENCLAW_GATEWAY__HOST"] = args.host
    if args.port:
        os.environ["OPENCLAW_GATEWAY__PORT"] = str(args.port)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nAu revoir !")
        sys.exit(0)


if __name__ == "__main__":
    main()
