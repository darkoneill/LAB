"""
Tests for openclaw/channels/telegram.py
Mocks python-telegram-bot entirely – no network calls.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw.channels.telegram import (
    TelegramChannel,
    escape_markdown_v2,
    normalise_message,
)


# ── Fixtures ────────────────────────────────────────────────


class FakeSettings:
    _base_dir = None

    def __init__(self, overrides=None):
        self._data = {
            "channels.telegram.token": "123456:ABC-TEST",
            "channels.telegram.allowed_users": [111, 222],
            "channels.telegram.session_scope": "per_user",
        }
        if overrides:
            self._data.update(overrides)

    def get(self, dotpath, default=None):
        return self._data.get(dotpath, default)


@pytest.fixture
def fake_settings():
    return FakeSettings()


@pytest.fixture
def brain():
    b = MagicMock()
    b.generate = AsyncMock(return_value={
        "content": "Bonjour !", "model": "mock", "usage": {}, "tool_calls": [],
    })
    return b


@pytest.fixture
def memory():
    m = MagicMock()
    m.search = AsyncMock(return_value=[])
    m.store_interaction = AsyncMock()
    return m


@pytest.fixture
def channel(fake_settings, brain, memory):
    with patch("openclaw.channels.telegram.get_settings", return_value=fake_settings):
        ch = TelegramChannel(brain=brain, memory_manager=memory)
    return ch


def _make_update(user_id=111, username="alice", text="hello", first_name="Alice"):
    """Build a fake telegram Update with message."""
    user = SimpleNamespace(id=user_id, username=username, first_name=first_name)
    message = MagicMock()
    message.text = text
    message.reply_text = AsyncMock()
    update = SimpleNamespace(effective_user=user, message=message)
    return update


# ══════════════════════════════════════════════════════════════
#  ESCAPE / NORMALISATION UTILITIES
# ══════════════════════════════════════════════════════════════


class TestEscapeMarkdownV2:

    def test_escapes_special_chars(self):
        assert escape_markdown_v2("hello_world") == r"hello\_world"
        assert escape_markdown_v2("a*b") == r"a\*b"
        assert escape_markdown_v2("[link](url)") == r"\[link\]\(url\)"

    def test_escapes_backticks(self):
        assert escape_markdown_v2("`code`") == r"\`code\`"

    def test_no_escape_plain_text(self):
        assert escape_markdown_v2("hello world 123") == "hello world 123"

    def test_multiple_special_chars(self):
        result = escape_markdown_v2("~#+-=|{}.!")
        assert "\\" in result
        # Each of the 10 chars should be escaped
        assert result.count("\\") == 10


class TestNormaliseMessage:

    def test_strips_whitespace(self):
        assert normalise_message("  hello  ") == "hello"

    def test_collapses_blank_lines(self):
        text = "line1\n\n\n\nline2"
        assert normalise_message(text) == "line1\n\nline2"

    def test_truncates_long_messages(self):
        text = "a" * 5000
        result = normalise_message(text)
        assert len(result) == 4000

    def test_preserves_single_newlines(self):
        assert normalise_message("a\nb") == "a\nb"

    def test_empty_message(self):
        assert normalise_message("   ") == ""


# ══════════════════════════════════════════════════════════════
#  ALLOWLIST (deny-by-default / ZeroClaw pattern)
# ══════════════════════════════════════════════════════════════


class TestAllowlist:

    def test_allowed_user_passes(self, channel):
        assert channel.is_allowed(111) is True
        assert channel.is_allowed(222) is True

    def test_unknown_user_denied(self, channel):
        assert channel.is_allowed(999) is False

    def test_empty_allowlist_denies_all(self, brain, memory):
        """Empty allowlist = deny everyone (ZeroClaw pattern)."""
        settings = FakeSettings({"channels.telegram.allowed_users": []})
        with patch("openclaw.channels.telegram.get_settings", return_value=settings):
            ch = TelegramChannel(brain=brain, memory_manager=memory)
        assert ch.is_allowed(111) is False
        assert ch.is_allowed(1) is False

    def test_allowlist_integer_coercion(self, brain, memory):
        """String user IDs in config are coerced to int."""
        settings = FakeSettings({"channels.telegram.allowed_users": ["333", "444"]})
        with patch("openclaw.channels.telegram.get_settings", return_value=settings):
            ch = TelegramChannel(brain=brain, memory_manager=memory)
        assert ch.is_allowed(333) is True
        assert ch.is_allowed(444) is True


# ══════════════════════════════════════════════════════════════
#  COMMAND HANDLERS
# ══════════════════════════════════════════════════════════════


class TestCommandStart:

    @pytest.mark.asyncio
    async def test_allowed_user_gets_welcome(self, channel):
        update = _make_update(user_id=111, first_name="Alice")
        await channel._cmd_start(update, None)
        update.message.reply_text.assert_called_once()
        text = update.message.reply_text.call_args[0][0]
        assert "Alice" in text
        assert "NexusMind" in text

    @pytest.mark.asyncio
    async def test_denied_user_gets_access_denied(self, channel):
        update = _make_update(user_id=999)
        await channel._cmd_start(update, None)
        update.message.reply_text.assert_called_once_with("Access denied.")


class TestCommandReset:

    @pytest.mark.asyncio
    async def test_reset_creates_new_session(self, channel):
        # Create an initial session with a message
        channel.sessions.get_or_create("tg_111")
        channel.sessions.add_message("tg_111", "user", "old message")
        assert len(channel.sessions.get_history("tg_111")) == 1

        update = _make_update(user_id=111)
        await channel._cmd_reset(update, None)

        # Session was recreated, old messages gone
        history = channel.sessions.get_history("tg_111")
        assert len(history) == 0
        update.message.reply_text.assert_called_once_with("Session réinitialisée.")

    @pytest.mark.asyncio
    async def test_reset_denied_for_unknown_user(self, channel):
        update = _make_update(user_id=999)
        await channel._cmd_reset(update, None)
        update.message.reply_text.assert_called_once_with("Access denied.")


class TestCommandStatus:

    @pytest.mark.asyncio
    async def test_status_shows_session_info(self, channel):
        update = _make_update(user_id=111)
        await channel._cmd_status(update, None)
        text = update.message.reply_text.call_args[0][0]
        assert "tg_111" in text
        assert "Telegram" in text

    @pytest.mark.asyncio
    async def test_status_denied_for_unknown_user(self, channel):
        update = _make_update(user_id=999)
        await channel._cmd_status(update, None)
        update.message.reply_text.assert_called_once_with("Access denied.")


class TestCommandHelp:

    @pytest.mark.asyncio
    async def test_help_lists_commands(self, channel):
        update = _make_update(user_id=111)
        await channel._cmd_help(update, None)
        text = update.message.reply_text.call_args[0][0]
        assert "/start" in text
        assert "/reset" in text
        assert "/status" in text
        assert "/help" in text

    @pytest.mark.asyncio
    async def test_help_denied_for_unknown_user(self, channel):
        update = _make_update(user_id=999)
        await channel._cmd_help(update, None)
        update.message.reply_text.assert_called_once_with("Access denied.")


# ══════════════════════════════════════════════════════════════
#  TEXT MESSAGE HANDLER
# ══════════════════════════════════════════════════════════════


class TestOnMessage:

    @pytest.mark.asyncio
    async def test_allowed_user_gets_reply(self, channel, brain):
        update = _make_update(user_id=111, text="Bonjour")
        await channel._on_message(update, None)

        brain.generate.assert_called_once()
        update.message.reply_text.assert_called_once_with("Bonjour !")

    @pytest.mark.asyncio
    async def test_denied_user_gets_access_denied(self, channel, brain):
        update = _make_update(user_id=999, text="hack")
        await channel._on_message(update, None)

        brain.generate.assert_not_called()
        update.message.reply_text.assert_called_once_with("Access denied.")

    @pytest.mark.asyncio
    async def test_message_stored_in_session(self, channel, brain):
        update = _make_update(user_id=111, text="test message")
        await channel._on_message(update, None)

        history = channel.sessions.get_history("tg_111")
        assert len(history) == 2  # user + assistant
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "test message"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Bonjour !"

    @pytest.mark.asyncio
    async def test_normalised_message_sent_to_brain(self, channel, brain):
        update = _make_update(user_id=111, text="  lots   of   spaces  ")
        await channel._on_message(update, None)

        call_kwargs = brain.generate.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[-1]["content"] == "lots   of   spaces"

    @pytest.mark.asyncio
    async def test_empty_message_ignored(self, channel, brain):
        update = _make_update(user_id=111, text="   ")
        await channel._on_message(update, None)
        brain.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_memory_search_called(self, channel, brain, memory):
        update = _make_update(user_id=111, text="remember me")
        await channel._on_message(update, None)

        memory.search.assert_called_once()
        memory.store_interaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_brain_error_returns_error_message(self, channel, brain):
        brain.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        update = _make_update(user_id=111, text="hello")
        await channel._on_message(update, None)

        reply = update.message.reply_text.call_args[0][0]
        assert "Erreur" in reply

    @pytest.mark.asyncio
    async def test_memory_failure_non_fatal(self, channel, brain, memory):
        """Memory search failure should not prevent the reply."""
        memory.search = AsyncMock(side_effect=Exception("db error"))
        update = _make_update(user_id=111, text="hello")
        await channel._on_message(update, None)

        # Brain should still be called and reply sent
        brain.generate.assert_called_once()
        update.message.reply_text.assert_called_once_with("Bonjour !")


# ══════════════════════════════════════════════════════════════
#  CHANNEL INTERFACE
# ══════════════════════════════════════════════════════════════


class TestChannelInterface:

    def test_name_property(self, channel):
        assert channel.name == "telegram"

    def test_implements_channel_base(self, channel):
        from openclaw.channels.base import ChannelBase
        assert isinstance(channel, ChannelBase)

    def test_session_id_format(self, channel):
        assert channel._session_id(111) == "tg_111"
        assert channel._session_id(999999) == "tg_999999"

    @pytest.mark.asyncio
    async def test_send_message_escapes_markdown(self, channel):
        """send_message escapes MarkdownV2 special chars."""
        channel._app = MagicMock()
        channel._app.bot.send_message = AsyncMock()

        await channel.send_message("111", "hello_world")
        call_kwargs = channel._app.bot.send_message.call_args.kwargs
        assert call_kwargs["text"] == r"hello\_world"
        assert call_kwargs["parse_mode"] == "MarkdownV2"

    @pytest.mark.asyncio
    async def test_send_message_noop_without_app(self, channel):
        """send_message does nothing if the app isn't started."""
        channel._app = None
        await channel.send_message("111", "hello")  # should not raise

    @pytest.mark.asyncio
    async def test_start_requires_token(self, brain, memory):
        """Starting without a token raises RuntimeError."""
        settings = FakeSettings({"channels.telegram.token": ""})
        with patch("openclaw.channels.telegram.get_settings", return_value=settings):
            ch = TelegramChannel(brain=brain, memory_manager=memory)
        with pytest.raises(RuntimeError, match="token not configured"):
            await ch.start()


# ══════════════════════════════════════════════════════════════
#  DEFAULT CONFIG
# ══════════════════════════════════════════════════════════════


class TestDefaultConfig:

    def test_telegram_disabled_by_default(self):
        """default.yaml should have telegram disabled."""
        import yaml
        from pathlib import Path

        default_yaml = Path(__file__).parent.parent / "openclaw" / "config" / "default.yaml"
        with open(default_yaml) as f:
            cfg = yaml.safe_load(f)

        tg = cfg["channels"]["telegram"]
        assert tg["enabled"] is False
        assert tg["token"] == ""
        assert tg["allowed_users"] == []
        assert tg["session_scope"] == "per_user"
