"""
Tests for the Discord channel adapter.
All network I/O and discord.py internals are mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openclaw.channels.discord import DiscordChannel, split_message


# ── split_message utility ──────────────────────────────────


class TestSplitMessage:
    def test_short_text_returns_single_chunk(self):
        assert split_message("hello") == ["hello"]

    def test_exact_limit_returns_single_chunk(self):
        text = "x" * 2000
        assert split_message(text) == [text]

    def test_splits_over_2000(self):
        text = "a" * 3000
        chunks = split_message(text)
        assert len(chunks) == 2
        assert all(len(c) <= 2000 for c in chunks)
        assert "".join(chunks) == text

    def test_prefers_newline_split(self):
        line1 = "a" * 1500
        line2 = "b" * 1500
        text = line1 + "\n" + line2
        chunks = split_message(text)
        assert chunks[0] == line1
        assert chunks[1] == line2

    def test_prefers_space_split_when_no_newline(self):
        word1 = "a" * 1500
        word2 = "b" * 1500
        text = word1 + " " + word2
        chunks = split_message(text)
        assert chunks[0] == word1
        assert chunks[1] == " " + word2 or len(chunks[1]) <= 2000

    def test_hard_cuts_when_no_whitespace(self):
        text = "x" * 5000
        chunks = split_message(text, limit=2000)
        assert all(len(c) <= 2000 for c in chunks)
        assert "".join(chunks) == text

    def test_custom_limit(self):
        text = "hello world foo bar"
        chunks = split_message(text, limit=11)
        assert all(len(c) <= 11 for c in chunks)

    def test_empty_text(self):
        assert split_message("") == [""]


# ── Access control (deny-by-default) ────────────────────────


class TestDiscordAccessControl:
    def _make_channel(self, allowed_users=None, allowed_guilds=None):
        """Create a DiscordChannel with mocked settings."""
        users = allowed_users or []
        guilds = allowed_guilds or []
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "channels.discord.token": "fake-token",
            "channels.discord.allowed_guilds": guilds,
            "channels.discord.allowed_users": users,
        }.get(k, d)

        with patch("openclaw.channels.discord.get_settings", return_value=settings):
            with patch("openclaw.channels.discord.SessionManager"):
                ch = DiscordChannel(brain=MagicMock(), memory_manager=None)
        return ch

    def test_deny_by_default_empty_lists(self):
        ch = self._make_channel(allowed_users=[], allowed_guilds=[])
        assert ch.is_allowed(12345, 99999) is False

    def test_deny_unknown_user(self):
        ch = self._make_channel(allowed_users=[111], allowed_guilds=[999])
        assert ch.is_allowed(222, 999) is False

    def test_deny_unknown_guild(self):
        ch = self._make_channel(allowed_users=[111], allowed_guilds=[999])
        assert ch.is_allowed(111, 888) is False

    def test_allow_valid_user_and_guild(self):
        ch = self._make_channel(allowed_users=[111], allowed_guilds=[999])
        assert ch.is_allowed(111, 999) is True

    def test_allow_dm_for_allowed_user(self):
        ch = self._make_channel(allowed_users=[111], allowed_guilds=[999])
        # DMs have guild_id=None — should be allowed if user is in list
        assert ch.is_allowed(111, None) is True

    def test_deny_dm_for_unknown_user(self):
        ch = self._make_channel(allowed_users=[111], allowed_guilds=[999])
        assert ch.is_allowed(222, None) is False

    def test_deny_guild_when_no_guilds_configured(self):
        ch = self._make_channel(allowed_users=[111], allowed_guilds=[])
        # User is allowed but no guilds are configured
        assert ch.is_allowed(111, 999) is False

    def test_multiple_users_and_guilds(self):
        ch = self._make_channel(
            allowed_users=[111, 222, 333],
            allowed_guilds=[900, 901],
        )
        assert ch.is_allowed(222, 901) is True
        assert ch.is_allowed(333, 900) is True
        assert ch.is_allowed(444, 900) is False
        assert ch.is_allowed(111, 902) is False


# ── Channel properties ──────────────────────────────────────


class TestDiscordChannelProperties:
    def _make_channel(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "channels.discord.token": "fake-token",
            "channels.discord.allowed_guilds": [],
            "channels.discord.allowed_users": [],
        }.get(k, d)
        with patch("openclaw.channels.discord.get_settings", return_value=settings):
            with patch("openclaw.channels.discord.SessionManager"):
                return DiscordChannel(brain=MagicMock())

    def test_name(self):
        ch = self._make_channel()
        assert ch.name == "discord"

    def test_session_id_format(self):
        assert DiscordChannel._session_id(12345) == "dc_12345"

    @pytest.mark.asyncio
    async def test_start_raises_without_token(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "channels.discord.token": "",
            "channels.discord.allowed_guilds": [],
            "channels.discord.allowed_users": [],
        }.get(k, d)
        with patch("openclaw.channels.discord.get_settings", return_value=settings):
            with patch("openclaw.channels.discord.SessionManager"):
                ch = DiscordChannel(brain=MagicMock())

        with pytest.raises(RuntimeError, match="Discord bot token not configured"):
            await ch.start()


# ── Message handling ────────────────────────────────────────


class TestDiscordHandleMessage:
    def _make_channel(self, allowed_users=None, allowed_guilds=None):
        users = allowed_users or [111]
        guilds = allowed_guilds or [999]
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "channels.discord.token": "fake-token",
            "channels.discord.allowed_guilds": guilds,
            "channels.discord.allowed_users": users,
        }.get(k, d)
        with patch("openclaw.channels.discord.get_settings", return_value=settings):
            ch = DiscordChannel(brain=AsyncMock(), memory_manager=None)
        return ch

    @pytest.mark.asyncio
    async def test_allowed_user_can_message(self):
        ch = self._make_channel(allowed_users=[111], allowed_guilds=[999])
        ch.brain.generate = AsyncMock(return_value={"content": "Reply!"})

        message = MagicMock()
        message.author.id = 111
        message.guild.id = 999
        message.content = "Hello bot"
        message.channel.send = AsyncMock()

        await ch._handle_message(message)

        message.channel.send.assert_called_once_with("Reply!")

    @pytest.mark.asyncio
    async def test_denied_user_gets_no_reply(self):
        ch = self._make_channel(allowed_users=[111], allowed_guilds=[999])

        message = MagicMock()
        message.author.id = 222  # Not in allowed_users
        message.guild.id = 999
        message.content = "Hello bot"
        message.channel.send = AsyncMock()

        await ch._handle_message(message)

        message.channel.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_message_ignored(self):
        ch = self._make_channel(allowed_users=[111], allowed_guilds=[999])

        message = MagicMock()
        message.author.id = 111
        message.guild.id = 999
        message.content = "   "
        message.channel.send = AsyncMock()

        await ch._handle_message(message)

        message.channel.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_reply_is_split(self):
        ch = self._make_channel(allowed_users=[111], allowed_guilds=[999])
        long_reply = "x" * 3500
        ch.brain.generate = AsyncMock(return_value={"content": long_reply})

        message = MagicMock()
        message.author.id = 111
        message.guild.id = 999
        message.content = "Give me a long answer"
        message.channel.send = AsyncMock()

        await ch._handle_message(message)

        # Should have been split into multiple sends
        assert message.channel.send.call_count == 2
        for call in message.channel.send.call_args_list:
            chunk = call[0][0]
            assert len(chunk) <= 2000


# ── Generate reply ──────────────────────────────────────────


class TestDiscordGenerateReply:
    def _make_channel(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "channels.discord.token": "fake-token",
            "channels.discord.allowed_guilds": [999],
            "channels.discord.allowed_users": [111],
        }.get(k, d)
        with patch("openclaw.channels.discord.get_settings", return_value=settings):
            ch = DiscordChannel(brain=AsyncMock(), memory_manager=None)
        return ch

    @pytest.mark.asyncio
    async def test_generate_reply_calls_brain(self):
        ch = self._make_channel()
        ch.brain.generate = AsyncMock(return_value={"content": "Bot reply"})

        reply = await ch._generate_reply(111, "Hello")

        assert reply == "Bot reply"
        ch.brain.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_reply_stores_in_session(self):
        ch = self._make_channel()
        ch.brain.generate = AsyncMock(return_value={"content": "Reply"})
        ch.sessions = MagicMock()
        ch.sessions.get_or_create.return_value = {"messages": []}
        ch.sessions.get_history.return_value = []

        await ch._generate_reply(111, "Test message")

        # Should have stored both user and assistant messages
        ch.sessions.add_message.assert_any_call("dc_111", "user", "Test message")
        ch.sessions.add_message.assert_any_call("dc_111", "assistant", "Reply")

    @pytest.mark.asyncio
    async def test_generate_reply_with_memory(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "channels.discord.token": "fake-token",
            "channels.discord.allowed_guilds": [999],
            "channels.discord.allowed_users": [111],
        }.get(k, d)
        mock_memory = AsyncMock()
        mock_memory.search = AsyncMock(return_value=[{"content": "past context"}])

        with patch("openclaw.channels.discord.get_settings", return_value=settings):
            ch = DiscordChannel(brain=AsyncMock(), memory_manager=mock_memory)

        ch.brain.generate = AsyncMock(return_value={"content": "Reply"})

        reply = await ch._generate_reply(111, "Hello")

        assert reply == "Reply"
        mock_memory.search.assert_called_once_with("Hello", top_k=5)
        mock_memory.store_interaction.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_reply_handles_brain_error(self):
        ch = self._make_channel()
        ch.brain.generate = AsyncMock(side_effect=RuntimeError("LLM down"))

        reply = await ch._generate_reply(111, "Hello")

        assert "Erreur" in reply
        assert "LLM down" in reply


# ── send_message ────────────────────────────────────────────


class TestDiscordSendMessage:
    @pytest.mark.asyncio
    async def test_send_message_to_channel(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "channels.discord.token": "fake-token",
            "channels.discord.allowed_guilds": [],
            "channels.discord.allowed_users": [],
        }.get(k, d)
        with patch("openclaw.channels.discord.get_settings", return_value=settings):
            with patch("openclaw.channels.discord.SessionManager"):
                ch = DiscordChannel(brain=MagicMock())

        mock_channel = AsyncMock()
        ch._bot = MagicMock()
        ch._bot.get_channel = MagicMock(return_value=mock_channel)

        await ch.send_message("12345", "Hello!")

        mock_channel.send.assert_called_once_with("Hello!")

    @pytest.mark.asyncio
    async def test_send_message_no_bot(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "channels.discord.token": "fake-token",
            "channels.discord.allowed_guilds": [],
            "channels.discord.allowed_users": [],
        }.get(k, d)
        with patch("openclaw.channels.discord.get_settings", return_value=settings):
            with patch("openclaw.channels.discord.SessionManager"):
                ch = DiscordChannel(brain=MagicMock())

        # _bot is None by default
        await ch.send_message("12345", "Hello!")  # Should not raise

    @pytest.mark.asyncio
    async def test_send_message_splits_long_content(self):
        settings = MagicMock()
        settings.get = lambda k, d=None: {
            "channels.discord.token": "fake-token",
            "channels.discord.allowed_guilds": [],
            "channels.discord.allowed_users": [],
        }.get(k, d)
        with patch("openclaw.channels.discord.get_settings", return_value=settings):
            with patch("openclaw.channels.discord.SessionManager"):
                ch = DiscordChannel(brain=MagicMock())

        mock_channel = AsyncMock()
        ch._bot = MagicMock()
        ch._bot.get_channel = MagicMock(return_value=mock_channel)

        long_text = "y" * 4500
        await ch.send_message("12345", long_text)

        assert mock_channel.send.call_count >= 3
