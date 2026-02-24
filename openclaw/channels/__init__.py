from openclaw.channels.base import ChannelBase

__all__ = ["ChannelBase"]

# TelegramChannel is imported lazily to avoid hard dependency on
# python-telegram-bot when the channel is not enabled.
