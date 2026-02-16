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
    
    if platform == MessagingPlatform.WHATSAPP:
        from whatsapp_updated import WhatsAppClient
        
        client = WhatsAppClient(
            access_token=Config.WHATSAPP_ACCESS_TOKEN,
            phone_number_id=Config.WHATSAPP_PHONE_NUMBER_ID,
            api_version=Config.WHATSAPP_API_VERSION
        )
        print(f"✓ Initialized WhatsApp messaging client")
        return client
    
    elif platform == MessagingPlatform.TELEGRAM:
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
