"""
OpenClaw Setup Wizard
Interactive, conversational setup that makes OpenClaw operational immediately.
The user can talk to the assistant to configure additional options.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich.markdown import Markdown

from openclaw.config.settings import Settings, get_settings

console = Console()


WIZARD_BANNER = r"""
[cyan bold]
   ___                    ____ _
  / _ \ _ __   ___ _ __  / ___| | __ ___      __
 | | | | '_ \ / _ \ '_ \| |   | |/ _` \ \ /\ / /
 | |_| | |_) |  __/ | | | |___| | (_| |\ V  V /
  \___/| .__/ \___|_| |_|\____|_|\__,_| \_/\_/
       |_|
[/cyan bold]
[bold]  Assistant de Configuration[/bold]
[dim]  v1.0.0 - NexusMind[/dim]
"""

STEPS = [
    "Bienvenue",
    "Provider LLM",
    "Options",
    "Verification",
    "Terminé",
]


class SetupWizard:
    """Interactive setup wizard for first-run configuration."""

    def __init__(self):
        self.settings = None
        self.config = {}

    async def run(self) -> bool:
        """Run the setup wizard. Returns True if setup was completed."""
        console.clear()
        console.print(WIZARD_BANNER)
        console.print()

        # Initialize settings
        base_dir = Path(__file__).parent
        self.settings = Settings.initialize(str(base_dir))

        # Step 1: Welcome
        self._step_welcome()

        # Step 2: LLM Provider
        provider_ok = self._step_provider()

        # Step 3: Options
        self._step_options()

        # Step 4: Verification
        self._step_verify()

        # Step 5: Done
        self._step_done()

        return provider_ok

    def _show_progress(self, current: int):
        """Show wizard progress bar."""
        parts = []
        for i, step in enumerate(STEPS):
            if i < current:
                parts.append(f"[green][OK][/green] {step}")
            elif i == current:
                parts.append(f"[cyan bold]>> {step}[/cyan bold]")
            else:
                parts.append(f"[dim]   {step}[/dim]")
        console.print(" | ".join(parts))
        console.print()

    def _step_welcome(self):
        """Welcome step."""
        self._show_progress(0)

        welcome_text = """
Bienvenue dans **OpenClaw** !

Je suis ton assistant IA personnel. Je vais t'aider a me configurer
en quelques etapes rapides. Apres ca, je serai operationnel et tu pourras
me parler pour affiner la configuration.

Ce qu'on va faire :
1. Configurer un **provider LLM** (le cerveau de l'IA)
2. Choisir quelques **options** de base
3. **Verifier** que tout fonctionne

Ca prend moins de 2 minutes. C'est parti !
"""
        console.print(Panel(Markdown(welcome_text), border_style="cyan", title="[bold]Bienvenue[/bold]"))
        console.print()
        Prompt.ask("[dim]Appuie sur Entree pour continuer[/dim]", default="")

    def _step_provider(self) -> bool:
        """LLM Provider configuration step."""
        console.print()
        self._show_progress(1)

        console.print(Panel(
            "[bold]Quel provider LLM veux-tu utiliser ?[/bold]\n\n"
            "Un provider LLM est le service qui fait tourner l'intelligence artificielle.\n"
            "Tu as besoin d'au moins un provider pour que je fonctionne.",
            border_style="cyan",
            title="Provider LLM",
        ))

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="bold", width=3)
        table.add_column("Provider", style="cyan")
        table.add_column("Description")
        table.add_column("Cle API ?")

        table.add_row("1", "Anthropic (Claude)", "Recommande - modeles Claude Sonnet/Opus", "Oui")
        table.add_row("2", "OpenAI (GPT)", "Modeles GPT-4o, GPT-4o-mini", "Oui")
        table.add_row("3", "Ollama (local)", "Modeles locaux - gratuit, pas de cle", "Non")
        table.add_row("4", "Custom (OpenAI-compat)", "Tout endpoint compatible OpenAI", "Oui")
        table.add_row("5", "Configurer plus tard", "Tu pourras me demander apres", "")

        console.print(table)
        console.print()

        choice = Prompt.ask(
            "Ton choix",
            choices=["1", "2", "3", "4", "5"],
            default="1",
        )

        provider_configured = False

        if choice == "1":
            provider_configured = self._configure_anthropic()
        elif choice == "2":
            provider_configured = self._configure_openai()
        elif choice == "3":
            provider_configured = self._configure_ollama()
        elif choice == "4":
            provider_configured = self._configure_custom()
        elif choice == "5":
            console.print("[yellow]OK, tu pourras me dire '/config' plus tard pour configurer.[/yellow]")

        return provider_configured

    def _configure_anthropic(self) -> bool:
        console.print()
        console.print("[bold]Configuration Anthropic (Claude)[/bold]")
        console.print("[dim]Tu peux obtenir ta cle API sur https://console.anthropic.com/[/dim]")
        console.print()

        # Check env var first
        env_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if env_key:
            console.print(f"[success]Cle API detectee dans l'environnement (***{env_key[-4:]})[/success]")
            if Confirm.ask("Utiliser cette cle ?", default=True):
                self.settings.set("providers.anthropic.enabled", True, persist=True)
                self.settings.set("providers.anthropic.api_key", env_key, persist=True)
                console.print("[success]Anthropic configure ![/success]")
                return True

        api_key = Prompt.ask("Cle API Anthropic (sk-ant-...)", password=True)
        if api_key and api_key.startswith("sk-"):
            self.settings.set("providers.anthropic.enabled", True, persist=True)
            self.settings.set("providers.anthropic.api_key", api_key, persist=True)
            console.print("[success]Anthropic configure ![/success]")
            return True

        console.print("[warning]Cle API invalide ou vide. Tu pourras la configurer plus tard.[/warning]")
        return False

    def _configure_openai(self) -> bool:
        console.print()
        console.print("[bold]Configuration OpenAI (GPT)[/bold]")
        console.print("[dim]Tu peux obtenir ta cle API sur https://platform.openai.com/[/dim]")
        console.print()

        env_key = os.environ.get("OPENAI_API_KEY", "")
        if env_key:
            console.print(f"[success]Cle API detectee dans l'environnement (***{env_key[-4:]})[/success]")
            if Confirm.ask("Utiliser cette cle ?", default=True):
                self.settings.set("providers.openai.enabled", True, persist=True)
                self.settings.set("providers.openai.api_key", env_key, persist=True)
                console.print("[success]OpenAI configure ![/success]")
                return True

        api_key = Prompt.ask("Cle API OpenAI (sk-...)", password=True)
        if api_key and api_key.startswith("sk-"):
            self.settings.set("providers.openai.enabled", True, persist=True)
            self.settings.set("providers.openai.api_key", api_key, persist=True)

            base_url = Prompt.ask("URL de base (laisser vide pour OpenAI officiel)", default="")
            if base_url:
                self.settings.set("providers.openai.base_url", base_url, persist=True)

            console.print("[success]OpenAI configure ![/success]")
            return True

        console.print("[warning]Cle API invalide ou vide.[/warning]")
        return False

    def _configure_ollama(self) -> bool:
        console.print()
        console.print("[bold]Configuration Ollama (local)[/bold]")
        console.print("[dim]Ollama execute des modeles localement. Aucune cle API necessaire.[/dim]")
        console.print("[dim]Installe Ollama: https://ollama.com/[/dim]")
        console.print()

        base_url = Prompt.ask("URL Ollama", default="http://localhost:11434")
        model = Prompt.ask("Modele par defaut", default="llama3.2")

        self.settings.set("providers.ollama.enabled", True, persist=True)
        self.settings.set("providers.ollama.base_url", base_url, persist=True)
        self.settings.set("providers.ollama.default_model", model, persist=True)

        console.print(f"[success]Ollama configure avec le modele {model} ![/success]")
        return True

    def _configure_custom(self) -> bool:
        console.print()
        console.print("[bold]Configuration Provider Custom[/bold]")
        console.print("[dim]Tout endpoint compatible avec l'API OpenAI fonctionne.[/dim]")
        console.print()

        base_url = Prompt.ask("URL de base de l'API")
        if not base_url:
            console.print("[warning]URL requise.[/warning]")
            return False

        api_key = Prompt.ask("Cle API (si necessaire)", password=True, default="")
        model = Prompt.ask("Nom du modele par defaut")

        self.settings.set("providers.custom.enabled", True, persist=True)
        self.settings.set("providers.custom.base_url", base_url, persist=True)
        self.settings.set("providers.custom.api_key", api_key, persist=True)
        self.settings.set("providers.custom.default_model", model, persist=True)

        console.print(f"[success]Provider custom configure ![/success]")
        return True

    def _step_options(self):
        """Configure basic options."""
        console.print()
        self._show_progress(2)

        console.print(Panel(
            "[bold]Options de base[/bold]\n\n"
            "Quelques reglages rapides. Tu pourras tout modifier apres.",
            border_style="cyan",
            title="Options",
        ))

        # Language
        lang = Prompt.ask("Langue de l'interface", choices=["fr", "en"], default="fr")
        self.settings.set("app.language", lang, persist=True)

        # Gateway port
        port = Prompt.ask("Port du gateway", default="18789")
        self.settings.set("gateway.port", int(port), persist=True)

        # Web UI
        web_enabled = Confirm.ask("Activer l'interface web ?", default=True)
        self.settings.set("ui.web.enabled", web_enabled, persist=True)

        # Memory
        memory_enabled = Confirm.ask("Activer la memoire persistante ?", default=True)
        self.settings.set("memory.enabled", memory_enabled, persist=True)

        # Shell access
        shell_enabled = Confirm.ask("Autoriser l'execution de commandes shell ?", default=True)
        self.settings.set("tools.shell.enabled", shell_enabled, persist=True)
        if shell_enabled:
            sandboxed = Confirm.ask("Mode sandbox (recommande) ?", default=True)
            self.settings.set("tools.shell.sandboxed", sandboxed, persist=True)

        console.print("[success]Options sauvegardees ![/success]")

    def _step_verify(self):
        """Verify configuration."""
        console.print()
        self._show_progress(3)

        console.print(Panel(
            "[bold]Verification de la configuration[/bold]",
            border_style="cyan",
            title="Verification",
        ))

        table = Table(show_header=True, header_style="bold")
        table.add_column("Composant", style="cyan")
        table.add_column("Status")
        table.add_column("Details", style="dim")

        # Check providers
        providers_ok = False
        for p in ["anthropic", "openai", "ollama", "custom"]:
            if self.settings.get(f"providers.{p}.enabled", False):
                providers_ok = True
                key = self.settings.get(f"providers.{p}.api_key", "")
                key_display = f"***{key[-4:]}" if key and len(key) > 4 else "env/none"
                table.add_row(
                    f"Provider {p}",
                    "[green]OK[/green]",
                    f"Modele: {self.settings.get(f'providers.{p}.default_model', '?')} | Cle: {key_display}",
                )

        if not providers_ok:
            table.add_row("Provider LLM", "[yellow]NON CONFIGURE[/yellow]", "Tu peux le configurer via /config")

        # Gateway
        gw_host = self.settings.get("gateway.host", "127.0.0.1")
        gw_port = self.settings.get("gateway.port", 18789)
        table.add_row("Gateway", "[green]OK[/green]", f"http://{gw_host}:{gw_port}")

        # Memory
        mem = self.settings.get("memory.enabled", True)
        table.add_row("Memoire", "[green]Activee[/green]" if mem else "[yellow]Desactivee[/yellow]", "3 couches (MemU)")

        # Web UI
        web = self.settings.get("ui.web.enabled", True)
        web_port = self.settings.get("ui.web.port", 18790)
        table.add_row("Interface Web", "[green]Activee[/green]" if web else "[dim]Desactivee[/dim]", f"Port {web_port}")

        # Shell
        shell = self.settings.get("tools.shell.enabled", True)
        sandboxed = self.settings.get("tools.shell.sandboxed", True)
        shell_detail = "sandbox" if sandboxed else "acces complet"
        table.add_row("Shell", "[green]Actif[/green]" if shell else "[dim]Desactive[/dim]", shell_detail)

        console.print(table)
        console.print()

    def _step_done(self):
        """Final step."""
        console.print()
        self._show_progress(4)

        done_text = """
## Configuration terminee !

**OpenClaw est pret.** Tu peux maintenant :

- **Parler avec moi** directement dans le terminal
- Taper `/help` pour voir toutes les commandes
- Taper `/config` pour modifier la configuration
- Acceder a l'**interface web** sur le port configure

### Options supplementaires

Parle-moi simplement pour configurer :
- Ajouter d'autres providers LLM
- Configurer des skills personnalises
- Ajuster les limites de securite
- Personnaliser le comportement de l'IA
- Et tout le reste...

**Je suis la pour t'aider.**
"""
        console.print(Panel(
            Markdown(done_text),
            border_style="green",
            title="[bold green]Pret ![/bold green]",
        ))
        console.print()
        Prompt.ask("[dim]Appuie sur Entree pour demarrer OpenClaw[/dim]", default="")


async def run_wizard() -> bool:
    """Convenience function to run the wizard."""
    wizard = SetupWizard()
    return await wizard.run()
