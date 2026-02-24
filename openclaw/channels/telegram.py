"""
Telegram channel adapter for NexusMind.

Bridges python-telegram-bot with AgentBrain, enforcing
a deny-by-default allowlist (ZeroClaw pattern).
"""

import logging
import re
import time

from openclaw.channels.base import ChannelBase
from openclaw.config.settings import get_settings
from openclaw.gateway.server import SessionManager

logger = logging.getLogger("openclaw.channels.telegram")

# ── MarkdownV2 escaping ─────────────────────────────────────

# Characters that must be escaped in Telegram MarkdownV2
_MDVE_SPECIAL = re.compile(r"([_*\[\]()~`>#+\-=|{}.!\\])")


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2 format."""
    return _MDVE_SPECIAL.sub(r"\\\1", text)


# ── Message normalisation ────────────────────────────────────

def normalise_message(text: str) -> str:
    """Normalise an incoming Telegram message before sending to the brain.

    * Strips leading/trailing whitespace
    * Collapses multiple blank lines into one
    * Limits length to 4 000 chars (Telegram max is 4 096)
    """
    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:4000]


# ── TelegramChannel ─────────────────────────────────────────

class TelegramChannel(ChannelBase):
    """Telegram bot channel implementing :class:`ChannelBase`.

    Configuration (from ``channels.telegram.*``):
        token           – Bot API token from @BotFather
        allowed_users   – List of integer user-IDs (deny-by-default)
        session_scope   – ``"per_user"`` (default)
    """

    def __init__(self, brain, memory_manager=None):
        self.settings = get_settings()
        self.brain = brain
        self.memory = memory_manager
        self.sessions = SessionManager()

        self._token: str = self.settings.get("channels.telegram.token", "")
        self._allowed: list[int] = [
            int(uid)
            for uid in self.settings.get("channels.telegram.allowed_users", [])
        ]
        self._app = None  # telegram.ext.Application (set in start())

    # ── ChannelBase interface ────────────────────────────

    @property
    def name(self) -> str:
        return "telegram"

    async def start(self) -> None:
        """Build the Telegram Application and start polling."""
        if not self._token:
            raise RuntimeError(
                "Telegram bot token not configured. "
                "Set channels.telegram.token in config."
            )

        from telegram.ext import (
            ApplicationBuilder,
            CommandHandler,
            MessageHandler,
            filters,
        )

        self._app = (
            ApplicationBuilder()
            .token(self._token)
            .build()
        )

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("reset", self._cmd_reset))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )

        logger.info(
            "Telegram channel starting (allowed_users=%s)",
            self._allowed or "NONE – all messages will be denied",
        )
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

    async def stop(self) -> None:
        """Gracefully shut down the Telegram bot."""
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram channel stopped")

    async def send_message(self, recipient: str, content: str) -> None:
        """Send a MarkdownV2-formatted message to a Telegram chat."""
        if not self._app:
            return
        safe = escape_markdown_v2(content)
        await self._app.bot.send_message(
            chat_id=int(recipient),
            text=safe,
            parse_mode="MarkdownV2",
        )
        logger.debug("Sent message to %s (%d chars)", recipient, len(content))

    # ── Allowlist check ──────────────────────────────────

    def is_allowed(self, user_id: int) -> bool:
        """Deny-by-default: user must be in the allowlist."""
        if not self._allowed:
            return False
        return user_id in self._allowed

    # ── Session helpers ──────────────────────────────────

    def _session_id(self, user_id: int) -> str:
        return f"tg_{user_id}"

    # ── Command handlers ─────────────────────────────────

    async def _cmd_start(self, update, context) -> None:
        user = update.effective_user
        logger.info("/start from user %s (id=%s)", user.username, user.id)

        if not self.is_allowed(user.id):
            await update.message.reply_text("Access denied.")
            return

        session = self.sessions.get_or_create(self._session_id(user.id))
        welcome = (
            f"Bienvenue {user.first_name} !\n"
            f"Je suis NexusMind, votre assistant IA.\n\n"
            f"Session : {session['id']}\n"
            f"Commandes : /help, /reset, /status"
        )
        await update.message.reply_text(welcome)

    async def _cmd_reset(self, update, context) -> None:
        user = update.effective_user
        logger.info("/reset from user %s (id=%s)", user.username, user.id)

        if not self.is_allowed(user.id):
            await update.message.reply_text("Access denied.")
            return

        sid = self._session_id(user.id)
        if sid in self.sessions.sessions:
            del self.sessions.sessions[sid]
        self.sessions.get_or_create(sid)
        await update.message.reply_text("Session réinitialisée.")

    async def _cmd_status(self, update, context) -> None:
        user = update.effective_user
        logger.info("/status from user %s (id=%s)", user.username, user.id)

        if not self.is_allowed(user.id):
            await update.message.reply_text("Access denied.")
            return

        sid = self._session_id(user.id)
        session = self.sessions.get_or_create(sid)
        msg_count = len(session.get("messages", []))
        active = self.sessions.active_count
        status_text = (
            f"Session : {sid}\n"
            f"Messages : {msg_count}\n"
            f"Sessions actives : {active}\n"
            f"Canal : Telegram"
        )
        await update.message.reply_text(status_text)

    async def _cmd_help(self, update, context) -> None:
        user = update.effective_user
        logger.info("/help from user %s (id=%s)", user.username, user.id)

        if not self.is_allowed(user.id):
            await update.message.reply_text("Access denied.")
            return

        help_text = (
            "Commandes disponibles :\n"
            "/start  – Message d'accueil\n"
            "/reset  – Réinitialiser la session\n"
            "/status – Informations système\n"
            "/help   – Afficher cette aide\n\n"
            "Envoyez un message texte pour discuter."
        )
        await update.message.reply_text(help_text)

    # ── Text message handler ─────────────────────────────

    async def _on_message(self, update, context) -> None:
        """Handle an incoming text message."""
        user = update.effective_user
        text = update.message.text or ""

        logger.info(
            "Message from %s (id=%s): %s",
            user.username, user.id, text[:80],
        )

        if not self.is_allowed(user.id):
            logger.warning("Denied message from user %s (id=%s)", user.username, user.id)
            await update.message.reply_text("Access denied.")
            return

        normalised = normalise_message(text)
        if not normalised:
            return

        sid = self._session_id(user.id)
        session = self.sessions.get_or_create(sid)
        self.sessions.add_message(sid, "user", normalised)

        # Build context and call brain
        context_messages = self.sessions.get_history(sid)
        memory_context = ""
        if self.memory:
            try:
                results = await self.memory.search(normalised, top_k=5)
                if results:
                    memory_context = "\n".join(
                        r.get("content", "") for r in results
                    )
            except Exception as e:
                logger.warning("Memory search failed: %s", e)

        try:
            result = await self.brain.generate(
                messages=context_messages,
                memory_context=memory_context,
            )
            reply = result.get("content", "Pas de réponse.")
        except Exception as e:
            logger.error("Brain generate failed: %s", e)
            reply = f"Erreur : {e}"

        # Store assistant response
        self.sessions.add_message(sid, "assistant", reply)

        # Store in memory
        if self.memory:
            try:
                await self.memory.store_interaction(
                    user_message=normalised,
                    assistant_response=reply,
                    session_id=sid,
                )
            except Exception as e:
                logger.warning("Memory store failed: %s", e)

        logger.info("Reply to %s (id=%s): %s", user.username, user.id, reply[:80])

        # Send the response (plain text, no MarkdownV2 for brain output
        # to avoid escaping issues with code blocks from the LLM)
        await update.message.reply_text(reply)
