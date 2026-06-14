"""
Messaging Client Factory
Creates the appropriate messaging client based on configuration
"""

from messaging_interface import MessagingClient
from config import Config, MessagingPlatform


def create_messaging_client() -> MessagingClient:
    """
    Create and return the configured messaging client
    
    Returns:
        MessagingClient instance (WhatsApp or Telegram)
    """
    
    platform = Config.get_messaging_platform()

    # WhatsApp support has been removed (legacy). Telegram remains only as a dormant
    # legacy delivery client; all active flows go through the iOS app + APNs.
    if platform == MessagingPlatform.TELEGRAM:
        from telegram import TelegramClient

        client = TelegramClient(
            bot_token=Config.TELEGRAM_BOT_TOKEN
        )
        print(f"✓ Initialized Telegram messaging client")
        return client

    else:
        raise ValueError(f"Unsupported messaging platform: {platform}")


def get_platform_name() -> str:
    """Get the name of the current messaging platform"""
    return Config.get_messaging_platform().value
