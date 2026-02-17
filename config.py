"""
Configuration Management
Centralized config for messaging platform selection
"""

import os
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


class MessagingPlatform(str, Enum):
    """Supported messaging platforms"""
    WHATSAPP = "whatsapp"
    TELEGRAM = "telegram"


class Config:
    """Application configuration"""
    
    # Messaging Platform Selection
    MESSAGING_PLATFORM = os.getenv("MESSAGING_PLATFORM", "telegram").lower()

    ENABLE_OTP = os.getenv("ENABLE_OTP", "false").lower() == "true"
    
    # WhatsApp Configuration
    WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
    WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "your_verify_token")
    WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v21.0")
    
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    TELEGRAM_WEBHOOK_URL = os.getenv("TELEGRAM_WEBHOOK_URL")
    
    # Cerebras AI
    CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
    
    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    # Application
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    @classmethod
    def get_messaging_platform(cls) -> MessagingPlatform:
        """Get the configured messaging platform"""
        platform = cls.MESSAGING_PLATFORM
        if platform == "whatsapp":
            return MessagingPlatform.WHATSAPP
        elif platform == "telegram":
            return MessagingPlatform.TELEGRAM
        else:
            raise ValueError(f"Unsupported messaging platform: {platform}")
    
    @classmethod
    def validate_config(cls):
        """Validate that required config is present"""
        platform = cls.get_messaging_platform()
        
        if platform == MessagingPlatform.WHATSAPP:
            if not cls.WHATSAPP_ACCESS_TOKEN or not cls.WHATSAPP_PHONE_NUMBER_ID:
                raise ValueError("WhatsApp config incomplete")
        
        elif platform == MessagingPlatform.TELEGRAM:
            if not cls.TELEGRAM_BOT_TOKEN:
                raise ValueError("Telegram config incomplete")
        
        if not cls.CEREBRAS_API_KEY:
            raise ValueError("Cerebras API key not set")
