"""
Messaging Platform Interface
Abstract interface for messaging clients (WhatsApp, Telegram, etc.)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class MessagingClient(ABC):
    """Abstract base class for messaging platform clients"""
    
    @abstractmethod
    async def send_message(
        self,
        to: str,
        message: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send text message
        
        Args:
            to: Recipient identifier (phone number or chat_id)
            message: Message text
            **kwargs: Platform-specific options
        
        Returns:
            Response from messaging API
        """
        pass
    
    @abstractmethod
    async def send_image(
        self,
        to: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send image message"""
        pass
    
    @abstractmethod
    async def send_document(
        self,
        to: str,
        document_url: str,
        filename: str,
        caption: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send document message"""
        pass
    
    @abstractmethod
    async def get_media_url(self, media_id: str) -> str:
        """Get media URL from media ID"""
        pass
    
    @abstractmethod
    async def download_media(self, media_url: str) -> bytes:
        """Download media file"""
        pass
    
    @abstractmethod
    def extract_user_identifier(self, webhook_data: Dict) -> Optional[str]:
        """
        Extract user identifier from webhook data
        
        Returns:
            User identifier (phone number for WhatsApp, chat_id for Telegram)
        """
        pass
    
    @abstractmethod
    def extract_message_data(self, webhook_data: Dict) -> List[Dict[str, Any]]:
        """
        Extract message data from webhook
        
        Returns:
            List of message dictionaries with keys:
            - user_id: User identifier
            - content: Message content
            - message_type: Type of message
            - media_url: Optional media URL
            - message_id: Platform message ID
        """
        pass
    
    @abstractmethod
    async def setup_webhook(self, webhook_url: str) -> Dict[str, Any]:
        """Setup webhook for receiving messages"""
        pass
    
    @abstractmethod
    async def send_buttons(
        self,
        to: str,
        message: str,
        buttons: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Send message with interactive buttons"""
        pass
    
    @abstractmethod
    async def get_webhook_info(self) -> Dict[str, Any]:
        """Get current webhook configuration/status"""
        pass
