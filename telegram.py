"""
Telegram Bot API Client
Handles sending/receiving messages via Telegram Bot API
"""

import os
import httpx
from typing import Optional, Dict, Any, List
from messaging_interface import MessagingClient


class TelegramClient(MessagingClient):
    """Client for Telegram Bot API"""
    
    def __init__(
        self,
        bot_token: str | None = None,
        api_base: str = "https://api.telegram.org"
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.bot_token:
            raise ValueError("Telegram bot token is required")
        
        self.api_base = api_base
        self.base_url = f"{api_base}/bot{self.bot_token}"
    
    async def send_message(
        self,
        to: str,
        message: str,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send text message to Telegram user
        
        Args:
            to: Chat ID
            message: Message text
            parse_mode: Text formatting (HTML, Markdown, MarkdownV2)
            disable_web_page_preview: Disable link previews
        
        Returns:
            Response from Telegram API
        """
        
        url = f"{self.base_url}/sendMessage"
        
        payload = {
            "chat_id": to,
            "text": message,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    async def send_image(
        self,
        to: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send photo message"""
        
        url = f"{self.base_url}/sendPhoto"
        
        payload = {
            "chat_id": to,
            "photo": image_url
        }
        
        if caption:
            payload["caption"] = caption
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    async def send_document(
        self,
        to: str,
        document_url: str,
        filename: str,
        caption: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send document message"""
        
        url = f"{self.base_url}/sendDocument"
        
        payload = {
            "chat_id": to,
            "document": document_url
        }
        
        if caption:
            payload["caption"] = caption
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    async def send_audio(
        self,
        to: str,
        audio_url: str,
        caption: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send audio message"""
        
        url = f"{self.base_url}/sendAudio"
        
        payload = {
            "chat_id": to,
            "audio": audio_url
        }
        
        if caption:
            payload["caption"] = caption
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    async def get_media_url(self, media_id: str) -> str:
        """
        Get file URL from Telegram file_id
        
        Args:
            media_id: File ID from Telegram message (Telegram calls this file_id)
        
        Returns:
            URL to download the file
        """
        
        url = f"{self.base_url}/getFile"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json={"file_id": media_id})
            response.raise_for_status()
            data = response.json()
            
            if data.get("ok"):
                file_path = data["result"]["file_path"]
                return f"{self.api_base}/file/bot{self.bot_token}/{file_path}"
            
            raise ValueError("Failed to get file URL")
    
    async def download_media(self, media_url: str) -> bytes:
        """
        Download media file from Telegram
        
        Args:
            media_url: URL from get_media_url()
        
        Returns:
            Media file content as bytes
        """
        
        async with httpx.AsyncClient() as client:
            response = await client.get(media_url)
            response.raise_for_status()
            return response.content
    
    async def setup_webhook(self, webhook_url: str) -> Dict[str, Any]:
        """
        Setup webhook for receiving messages
        
        Args:
            webhook_url: Your server's webhook URL
        
        Returns:
            Response from Telegram API
        """
        
        url = f"{self.base_url}/setWebhook"
        
        payload = {
            "url": webhook_url,
            "allowed_updates": ["message", "edited_message"]
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    async def delete_webhook(self) -> Dict[str, Any]:
        """Delete webhook (useful for development)"""
        
        url = f"{self.base_url}/deleteWebhook"
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url)
            response.raise_for_status()
            return response.json()
    
    async def get_webhook_info(self) -> Dict[str, Any]:
        """Get current webhook status"""
        
        url = f"{self.base_url}/getWebhookInfo"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()
    
    def extract_user_identifier(self, webhook_data: Dict) -> Optional[str]:
        """Extract chat_id from Telegram webhook"""
        
        message = webhook_data.get("message") or webhook_data.get("edited_message")
        if message:
            chat = message.get("chat", {})
            return str(chat.get("id"))
        
        return None
    
    def extract_message_data(self, webhook_data: Dict) -> List[Dict[str, Any]]:
        """Extract message data from Telegram webhook"""
        
        messages = []
        message = webhook_data.get("message") or webhook_data.get("edited_message")
        
        if not message:
            return messages
        
        chat_id = str(message["chat"]["id"])
        message_id = str(message["message_id"])
        
        # Text message
        if "text" in message:
            messages.append({
                "user_id": chat_id,
                "content": message["text"],
                "message_type": "text",
                "media_url": None,
                "message_id": message_id,
                "metadata": {
                    "username": message.get("from", {}).get("username"),
                    "first_name": message.get("from", {}).get("first_name"),
                    "last_name": message.get("from", {}).get("last_name")
                }
            })
        
        # Photo message
        elif "photo" in message:
            # Get the largest photo
            photos = message["photo"]
            largest_photo = max(photos, key=lambda p: p.get("file_size", 0))
            
            messages.append({
                "user_id": chat_id,
                "content": message.get("caption", "[Image]"),
                "message_type": "image",
                "media_url": None,  # Will be fetched using file_id
                "message_id": message_id,
                "metadata": {
                    "file_id": largest_photo["file_id"],
                    "file_size": largest_photo.get("file_size")
                }
            })
        
        # Document message
        elif "document" in message:
            doc = message["document"]
            
            messages.append({
                "user_id": chat_id,
                "content": message.get("caption", f"[Document: {doc.get('file_name', 'unknown')}]"),
                "message_type": "document",
                "media_url": None,
                "message_id": message_id,
                "metadata": {
                    "file_id": doc["file_id"],
                    "file_name": doc.get("file_name"),
                    "file_size": doc.get("file_size"),
                    "mime_type": doc.get("mime_type")
                }
            })
        
        # Audio/Voice message
        elif "audio" in message or "voice" in message:
            audio_data = message.get("audio") or message.get("voice")
            
            messages.append({
                "user_id": chat_id,
                "content": "[Audio]",
                "message_type": "audio",
                "media_url": None,
                "message_id": message_id,
                "metadata": {
                    "file_id": audio_data["file_id"],
                    "duration": audio_data.get("duration"),
                    "mime_type": audio_data.get("mime_type")
                }
            })
        
        # Video message
        elif "video" in message:
            video = message["video"]
            
            messages.append({
                "user_id": chat_id,
                "content": message.get("caption", "[Video]"),
                "message_type": "video",
                "media_url": None,
                "message_id": message_id,
                "metadata": {
                    "file_id": video["file_id"],
                    "duration": video.get("duration"),
                    "file_size": video.get("file_size")
                }
            })
        
        return messages
    
    async def send_buttons(
        self,
        to: str,
        message: str,
        buttons: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send message with inline keyboard buttons
        
        Args:
            to: Chat ID
            message: Message text
            buttons: List of button dicts [{"text": "Button 1", "callback_data": "btn1"}]
        """
        
        url = f"{self.base_url}/sendMessage"
        
        # Create inline keyboard
        keyboard = {
            "inline_keyboard": [
                [{"text": btn.get("text", btn.get("title", "Button")), 
                  "callback_data": btn.get("callback_data", btn.get("id", "btn"))}]
                for btn in buttons
            ]
        }
        
        payload = {
            "chat_id": to,
            "text": message,
            "reply_markup": keyboard
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    async def answer_callback_query(
        self,
        callback_query_id: str,
        text: Optional[str] = None,
        show_alert: bool = False
    ) -> Dict[str, Any]:
        """Answer callback query from inline button"""
        
        url = f"{self.base_url}/answerCallbackQuery"
        
        payload: Dict[str, Any] = {
            "callback_query_id": callback_query_id,
            "show_alert": show_alert
        }
        
        if text:
            payload["text"] = text
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
