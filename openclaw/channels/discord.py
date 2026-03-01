"""
Discord channel adapter for NexusMind.

Bridges discord.py with AgentBrain, enforcing deny-by-default
allowlists for both guilds and users (ZeroClaw pattern).
"""

import logging
import re

from openclaw.channels.base import ChannelBase
from openclaw.config.settings import get_settings
from openclaw.gateway.server import SessionManager

logger = logging.getLogger("openclaw.channels.discord")

# Discord message length limit
_MAX_MESSAGE_LEN = 2000


def split_message(text: str, limit: int = _MAX_MESSAGE_LEN) -> list[str]:
    """Split *text* into chunks of at most *limit* characters.

    Tries to break at newlines first, then at spaces, then hard-cuts.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break

        # Prefer splitting at last newline within limit
        cut = text.rfind("\n", 0, limit)
        if cut <= 0:
            cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit

        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


class DiscordChannel(ChannelBase):
    """Discord bot channel implementing :class:`ChannelBase`.

    Configuration (from ``channels.discord.*``):
        token           – Bot token from Discord Developer Portal
        allowed_guilds  – List of integer guild IDs (deny-by-default)
        allowed_users   – List of integer user IDs (deny-by-default)
    """

    def __init__(self, brain, memory_manager=None):
        self.settings = get_settings()
        self.brain = brain
        self.memory = memory_manager
        self.sessions = SessionManager()

        self._token: str = self.settings.get("channels.discord.token", "")
        self._allowed_guilds: list[int] = [
            int(gid)
            for gid in self.settings.get("channels.discord.allowed_guilds", [])
        ]
        self._allowed_users: list[int] = [
            int(uid)
            for uid in self.settings.get("channels.discord.allowed_users", [])
        ]
        self._bot = None  # discord.ext.commands.Bot (set in start())

    # ── ChannelBase interface ────────────────────────────

    @property
    def name(self) -> str:
        return "discord"

    async def start(self) -> None:
        if not self._token:
            raise RuntimeError(
                "Discord bot token not configured. "
                "Set channels.discord.token in config."
            )

        import discord
        from discord.ext import commands

        intents = discord.Intents.default()
        intents.message_content = True

        self._bot = commands.Bot(command_prefix="!", intents=intents)
        self._register_events()
        self._register_commands()

        logger.info(
            "Discord channel starting (allowed_guilds=%s, allowed_users=%s)",
            self._allowed_guilds or "NONE",
            self._allowed_users or "NONE",
        )
        await self._bot.start(self._token)

    async def stop(self) -> None:
        if self._bot:
            await self._bot.close()
            logger.info("Discord channel stopped")

    async def send_message(self, recipient: str, content: str) -> None:
        if not self._bot:
            return
        channel = self._bot.get_channel(int(recipient))
        if channel:
            for chunk in split_message(content):
                await channel.send(chunk)

    # ── Access control ───────────────────────────────────

    def is_allowed(self, user_id: int, guild_id: int | None) -> bool:
        """Deny-by-default: user must be in allowed_users AND
        (message is a DM OR guild is in allowed_guilds).
        """
        if not self._allowed_users:
            return False
        if user_id not in self._allowed_users:
            return False
        # DMs have no guild — allowed if user is in the list
        if guild_id is None:
            return True
        if not self._allowed_guilds:
            return False
        return guild_id in self._allowed_guilds

    # ── Session helpers ──────────────────────────────────

    @staticmethod
    def _session_id(user_id: int) -> str:
        return f"dc_{user_id}"

    # ── Event / command registration ─────────────────────

    def _register_events(self):
        bot = self._bot

        @bot.event
        async def on_ready():
            logger.info("Discord bot logged in as %s", bot.user)
            try:
                synced = await bot.tree.sync()
                logger.info("Synced %d slash command(s)", len(synced))
            except Exception as e:
                logger.warning("Slash command sync failed: %s", e)

        @bot.event
        async def on_message(message):
            # Ignore own messages
            if message.author == bot.user:
                return
            # Process commands first (slash & prefix)
            await bot.process_commands(message)
            # Skip command invocations
            ctx = await bot.get_context(message)
            if ctx.valid:
                return
            await self._handle_message(message)

    def _register_commands(self):
        bot = self._bot

        @bot.tree.command(name="ask", description="Poser une question a NexusMind")
        async def slash_ask(interaction, message: str):
            user = interaction.user
            guild_id = interaction.guild.id if interaction.guild else None
            if not self.is_allowed(user.id, guild_id):
                await interaction.response.send_message("Access denied.", ephemeral=True)
                return
            await interaction.response.defer()
            reply = await self._generate_reply(user.id, message)
            for chunk in split_message(reply):
                await interaction.followup.send(chunk)

        @bot.tree.command(name="reset", description="Reinitialiser la session")
        async def slash_reset(interaction):
            user = interaction.user
            guild_id = interaction.guild.id if interaction.guild else None
            if not self.is_allowed(user.id, guild_id):
                await interaction.response.send_message("Access denied.", ephemeral=True)
                return
            sid = self._session_id(user.id)
            if sid in self.sessions.sessions:
                del self.sessions.sessions[sid]
            self.sessions.get_or_create(sid)
            await interaction.response.send_message("Session reinitialised.")

        @bot.tree.command(name="status", description="Informations session")
        async def slash_status(interaction):
            user = interaction.user
            guild_id = interaction.guild.id if interaction.guild else None
            if not self.is_allowed(user.id, guild_id):
                await interaction.response.send_message("Access denied.", ephemeral=True)
                return
            sid = self._session_id(user.id)
            session = self.sessions.get_or_create(sid)
            msg_count = len(session.get("messages", []))
            await interaction.response.send_message(
                f"Session : {sid}\n"
                f"Messages : {msg_count}\n"
                f"Canal : Discord"
            )

    # ── Message handling ─────────────────────────────────

    async def _handle_message(self, message):
        """Handle a non-command text message."""
        user = message.author
        guild_id = message.guild.id if message.guild else None

        if not self.is_allowed(user.id, guild_id):
            return  # Silently ignore (no "access denied" on every message)

        text = message.content.strip()
        if not text:
            return

        reply = await self._generate_reply(user.id, text)
        for chunk in split_message(reply):
            await message.channel.send(chunk)

    async def _generate_reply(self, user_id: int, text: str) -> str:
        """Call brain.generate() and return the reply string."""
        sid = self._session_id(user_id)
        session = self.sessions.get_or_create(sid)
        self.sessions.add_message(sid, "user", text)

        context_messages = self.sessions.get_history(sid)
        memory_context = ""
        if self.memory:
            try:
                results = await self.memory.search(text, top_k=5)
                if results:
                    memory_context = "\n".join(r.get("content", "") for r in results)
            except Exception as e:
                logger.warning("Memory search failed: %s", e)

        try:
            result = await self.brain.generate(
                messages=context_messages,
                memory_context=memory_context,
            )
            reply = result.get("content", "Pas de reponse.")
        except Exception as e:
            logger.error("Brain generate failed: %s", e)
            reply = f"Erreur : {e}"

        self.sessions.add_message(sid, "assistant", reply)

        if self.memory:
            try:
                await self.memory.store_interaction(
                    user_message=text,
                    assistant_response=reply,
                    session_id=sid,
                )
            except Exception as e:
                logger.warning("Memory store failed: %s", e)

        return reply
