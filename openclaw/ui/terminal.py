"""
Terminal UI - Rich-based interactive terminal interface.
"""

import asyncio
import logging
import sys
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from openclaw.config.settings import get_settings

logger = logging.getLogger("openclaw.ui.terminal")

OPENCLAW_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red bold",
    "success": "green",
    "primary": "cyan bold",
    "secondary": "magenta",
    "muted": "dim",
    "user": "green bold",
    "assistant": "cyan",
    "system": "yellow dim",
    "tool": "magenta dim",
})

BANNER = r"""
[cyan]
   ___                    ____ _
  / _ \ _ __   ___ _ __  / ___| | __ ___      __
 | | | | '_ \ / _ \ '_ \| |   | |/ _` \ \ /\ / /
 | |_| | |_) |  __/ | | | |___| | (_| |\ V  V /
  \___/| .__/ \___|_| |_|\____|_|\__,_| \_/\_/
       |_|
[/cyan]
[dim]  NexusMind v1.0.0 - Autonomous AI Assistant[/dim]
[dim]  Powered by OpenClaw + MemU + AgentZero[/dim]
"""


class TerminalUI:
    """Interactive terminal interface with Rich rendering."""

    def __init__(self, agent_brain=None, memory_manager=None, orchestrator=None):
        self.settings = get_settings()
        self.console = Console(theme=OPENCLAW_THEME)
        self.agent = agent_brain
        self.memory = memory_manager
        self.orchestrator = orchestrator
        self.session_messages: list[dict] = []
        self.running = False

    def show_banner(self):
        """Display the startup banner."""
        self.console.print(BANNER)
        self.console.print()

    def show_status(self):
        """Show current status info."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("key", style="dim")
        table.add_column("value", style="cyan")

        providers = []
        for p in ["anthropic", "openai", "ollama", "custom"]:
            if self.settings.get(f"providers.{p}.enabled", False):
                providers.append(p)

        table.add_row("Providers", ", ".join(providers) if providers else "aucun (configuration requise)")
        table.add_row("Memory", "active" if self.settings.get("memory.enabled") else "disabled")
        table.add_row("Gateway", f"http://{self.settings.get('gateway.host')}:{self.settings.get('gateway.port')}")

        self.console.print(Panel(table, title="[primary]Status[/primary]", border_style="cyan"))

    def show_help(self):
        """Display help information."""
        help_text = """
## Commandes

| Commande | Description |
|----------|-------------|
| `/help` | Afficher cette aide |
| `/status` | Afficher le statut du systeme |
| `/config` | Modifier la configuration |
| `/memory` | Explorer la memoire |
| `/skills` | Lister les skills disponibles |
| `/tools` | Lister les outils |
| `/clear` | Effacer l'ecran |
| `/reset` | Reinitialiser la session |
| `/wizard` | Relancer le wizard de configuration |
| `/quit` | Quitter OpenClaw |

## Raccourcis

- **Ctrl+C** : Interrompre l'action en cours
- **Ctrl+D** : Quitter

## Usage

Parle-moi naturellement. Je suis ton assistant IA polyvalent.
"""
        self.console.print(Markdown(help_text))

    async def run(self):
        """Main interactive loop."""
        self.running = True
        self.show_banner()
        self.show_status()
        self.console.print()
        self.console.print("[dim]Tape /help pour l'aide, /quit pour quitter[/dim]")
        self.console.print()

        while self.running:
            try:
                user_input = await self._get_input()
                if user_input is None:
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                    continue

                # Chat interaction
                await self._handle_chat(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[warning]Interrompu.[/warning]")
                continue
            except EOFError:
                break

        self.console.print("\n[dim]Au revoir ![/dim]")

    async def _get_input(self) -> Optional[str]:
        """Get user input with prompt."""
        try:
            prompt_style = self.settings.get("ui.terminal.prompt_style", "arrow")
            prompts = {
                "arrow": "[user]>[/user] ",
                "lambda": "[user]lambda[/user] ",
                "caret": "[user]^[/user] ",
            }
            prompt_text = prompts.get(prompt_style, "[user]>[/user] ")

            # Use asyncio-compatible input
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self.console.input(prompt_text)
            )
        except (EOFError, KeyboardInterrupt):
            return None

    async def _handle_command(self, command: str):
        """Process a slash command."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "/quit" or cmd == "/exit":
            self.running = False
        elif cmd == "/help":
            self.show_help()
        elif cmd == "/status":
            self.show_status()
        elif cmd == "/clear":
            self.console.clear()
            self.show_banner()
        elif cmd == "/reset":
            self.session_messages.clear()
            if self.orchestrator:
                self.orchestrator.reset()
            self.console.print("[success]Session reinitialisee.[/success]")
        elif cmd == "/memory":
            await self._show_memory(args)
        elif cmd == "/skills":
            self._show_skills()
        elif cmd == "/tools":
            self._show_tools()
        elif cmd == "/config":
            await self._config_command(args)
        elif cmd == "/wizard":
            from openclaw.setup_wizard import SetupWizard
            wizard = SetupWizard()
            await wizard.run()
        else:
            self.console.print(f"[warning]Commande inconnue: {cmd}[/warning]")

    async def _handle_chat(self, message: str):
        """Handle a regular chat message."""
        self.session_messages.append({"role": "user", "content": message})

        if not self.agent:
            self.console.print(
                "[error]Agent non initialise. Lance /wizard pour configurer un provider LLM.[/error]"
            )
            return

        # Get memory context
        memory_context = ""
        if self.memory:
            memory_context = await self.memory.get_context_for_prompt(message)

        # Show thinking indicator
        self.console.print()
        with self.console.status("[dim]Reflexion...[/dim]", spinner="dots"):
            try:
                if self.orchestrator:
                    result = await self.orchestrator.process_message(
                        message,
                        session_messages=self.session_messages[:-1],  # Exclude last (already passed)
                        memory_context=memory_context,
                    )
                else:
                    result = await self.agent.generate(
                        messages=self.session_messages,
                        memory_context=memory_context,
                    )
            except Exception as e:
                self.console.print(f"[error]Erreur: {e}[/error]")
                return

        content = result.get("content", "")
        model = result.get("model", "")
        usage = result.get("usage", {})

        self.session_messages.append({"role": "assistant", "content": content})

        # Display response
        self.console.print()
        self.console.print(Markdown(content))

        # Show metadata
        if self.settings.get("ui.terminal.show_thinking", True) and usage:
            tokens_in = usage.get("input_tokens", 0)
            tokens_out = usage.get("output_tokens", 0)
            meta = f"[dim]{model} | {tokens_in}+{tokens_out} tokens[/dim]"
            self.console.print(meta)
        self.console.print()

        # Store in memory
        if self.memory:
            try:
                await self.memory.store_interaction(
                    user_message=message,
                    assistant_response=content,
                )
            except Exception as e:
                logger.debug(f"Memory storage error: {e}")

    async def _show_memory(self, args: str):
        """Show memory information."""
        if not self.memory:
            self.console.print("[warning]Systeme de memoire non actif.[/warning]")
            return

        if args:
            # Search memory
            results = await self.memory.search(args, top_k=10)
            if results:
                table = Table(title=f"Resultats memoire: '{args}'")
                table.add_column("Categorie", style="cyan")
                table.add_column("Contenu")
                table.add_column("Score", style="dim")
                for r in results:
                    table.add_row(
                        r.get("category", "?"),
                        r.get("content", "")[:80],
                        f"{r.get('significance', 0):.2f}",
                    )
                self.console.print(table)
            else:
                self.console.print("[dim]Aucun resultat.[/dim]")
        else:
            # Show categories
            categories = await self.memory.list_categories()
            stats = await self.memory.get_stats()

            table = Table(title="Memoire OpenClaw")
            table.add_column("Categorie", style="cyan")
            table.add_column("Elements", style="green", justify="right")
            table.add_column("Description", style="dim")
            for cat in categories:
                if cat.get("item_count", 0) > 0:
                    table.add_row(
                        cat["name"],
                        str(cat["item_count"]),
                        cat.get("description", ""),
                    )
            self.console.print(table)
            self.console.print(f"[dim]Total: {stats.get('resources', 0)} ressources, {stats.get('items', 0)} elements[/dim]")

    def _show_skills(self):
        """Display available skills."""
        self.console.print("[primary]Skills disponibles :[/primary]")
        skills = []
        # Try to get from config paths
        base = self.settings._base_dir
        if base:
            builtin = base / "skills" / "builtin"
            if builtin.exists():
                for d in builtin.iterdir():
                    if d.is_dir() and (d / "SKILL.md").exists():
                        skills.append(d.name)

        if skills:
            table = Table(show_header=True)
            table.add_column("Skill", style="cyan")
            table.add_column("Status", style="green")
            for s in skills:
                table.add_row(s, "active")
            self.console.print(table)
        else:
            self.console.print("[dim]Aucun skill charge.[/dim]")

    def _show_tools(self):
        """Display available tools."""
        tools = ["shell", "read_file", "write_file", "search_files"]
        table = Table(title="Outils disponibles")
        table.add_column("Outil", style="cyan")
        table.add_column("Status", style="green")
        for t in tools:
            enabled = self.settings.get(f"tools.{t.split('_')[0]}.enabled", True)
            table.add_row(t, "actif" if enabled else "desactive")
        self.console.print(table)

    async def _config_command(self, args: str):
        """Handle configuration commands."""
        if not args:
            self.console.print("[primary]Configuration actuelle :[/primary]")
            cfg = self.settings.all()
            # Show key sections
            for section in ["app", "gateway", "agent", "memory"]:
                if section in cfg:
                    self.console.print(f"\n[cyan]{section}:[/cyan]")
                    for k, v in cfg[section].items():
                        if not isinstance(v, dict):
                            self.console.print(f"  {k}: {v}")
        else:
            parts = args.split("=", 1)
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                # Auto-cast
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                self.settings.set(key, value, persist=True)
                self.console.print(f"[success]{key} = {value} (sauvegarde)[/success]")
            else:
                value = self.settings.get(args.strip())
                self.console.print(f"[cyan]{args.strip()}[/cyan] = {value}")
