"""
WhatsApp Business API Client
Handles sending/receiving messages via WhatsApp Cloud API
Updated to implement MessagingClient interface
"""

import os
import httpx
from typing import Optional, Dict, Any, List
from messaging_interface import MessagingClient


class WhatsAppClient(MessagingClient):
    """Client for WhatsApp Business Cloud API"""
    
    def __init__(
        self,
        access_token: str | None = None,
        phone_number_id: str | None = None,
        api_version: str = "v21.0"
    ):
        self.access_token = access_token or os.getenv("WHATSAPP_ACCESS_TOKEN")
        self.phone_number_id = phone_number_id or os.getenv("WHATSAPP_PHONE_NUMBER_ID")
        self.api_version = api_version
        self.base_url = f"https://graph.facebook.com/{api_version}"
    
    async def send_message(
        self,
        to: str,
        message: str,
        preview_url: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Send text message to WhatsApp user"""
        
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to,
            "type": "text",
            "text": {
                "preview_url": preview_url,
                "body": message
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    
    async def send_image(
        self,
        to: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send image message"""
        
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "image",
            "image": {
                "link": image_url
            }
        }
        
        if caption:
            payload["image"]["caption"] = caption
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
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
        
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "document",
            "document": {
                "link": document_url,
                "filename": filename
            }
        }
        
        if caption:
            payload["document"]["caption"] = caption
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    
    async def get_media_url(self, media_id: str) -> str:
        """Get media URL from media ID"""
        
        url = f"{self.base_url}/{media_id}"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("url", "")
    
    async def download_media(self, media_url: str) -> bytes:
        """Download media file from WhatsApp"""
        
        headers = {
            "Authorization": f"Bearer {self.access_token}"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(media_url, headers=headers)
            response.raise_for_status()
            return response.content
    
    def extract_user_identifier(self, webhook_data: Dict) -> Optional[str]:
        """Extract phone number from WhatsApp webhook"""
        
        for entry in webhook_data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                
                if messages:
                    return messages[0].get("from")
        
        return None
    
    def extract_message_data(self, webhook_data: Dict) -> List[Dict[str, Any]]:
        """Extract message data from WhatsApp webhook"""
        
        messages = []
        
        for entry in webhook_data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                
                for msg in value.get("messages", []):
                    phone = msg.get("from")
                    message_type = msg.get("type")
                    message_id = msg.get("id")
                    
                    msg_data = {
                        "user_id": phone,
                        "content": "",
                        "message_type": message_type,
                        "media_url": None,
                        "message_id": message_id,
                        "metadata": {}
                    }
                    
                    # Extract content based on type
                    if message_type == "text":
                        msg_data["content"] = msg.get("text", {}).get("body", "")
                    
                    elif message_type == "image":
                        msg_data["content"] = msg.get("image", {}).get("caption", "[Image]")
                        msg_data["metadata"]["media_id"] = msg.get("image", {}).get("id")
                    
                    elif message_type == "audio":
                        msg_data["content"] = "[Audio]"
                        msg_data["metadata"]["media_id"] = msg.get("audio", {}).get("id")
                    
                    elif message_type == "document":
                        doc = msg.get("document", {})
                        msg_data["content"] = f"[Document: {doc.get('filename', 'unknown')}]"
                        msg_data["metadata"]["media_id"] = doc.get("id")
                        msg_data["metadata"]["filename"] = doc.get("filename")
                    
                    elif message_type == "video":
                        msg_data["content"] = msg.get("video", {}).get("caption", "[Video]")
                        msg_data["metadata"]["media_id"] = msg.get("video", {}).get("id")
                    
                    messages.append(msg_data)
        
        return messages
    
    async def setup_webhook(self, webhook_url: str) -> Dict[str, Any]:
        """
        Setup webhook (Note: For WhatsApp, this is typically done via Meta dashboard)
        This method returns info about how to set it up
        """
        return {
            "platform": "whatsapp",
            "message": "WhatsApp webhooks must be configured via Meta Business Suite",
            "webhook_url": webhook_url,
            "verify_token": os.getenv("WHATSAPP_VERIFY_TOKEN"),
            "instructions": "Go to Meta Business Suite > WhatsApp > Configuration > Webhooks"
        }
    
    async def get_webhook_info(self) -> Dict[str, Any]:
        """
        Get webhook configuration info
        Note: WhatsApp doesn't provide an API to query webhook status,
        so we return the expected configuration
        """
        return {
            "platform": "whatsapp",
            "message": "WhatsApp webhook info must be checked via Meta Business Suite",
            "expected_url": "https://yourdomain.com/webhook/whatsapp",
            "verify_token": os.getenv("WHATSAPP_VERIFY_TOKEN"),
            "instructions": "Check Meta Business Suite > WhatsApp > Configuration > Webhooks"
        }
    
    async def send_buttons(
        self,
        to: str,
        message: str,
        buttons: List[Dict[str, str]],
        header_text: Optional[str] = None,
        footer_text: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Send interactive message with buttons"""
        
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        
        interactive = {
            "type": "button",
            "body": {"text": message},
            "action": {
                "buttons": [
                    {"type": "reply", "reply": {"id": btn.get("id", str(i)), "title": btn.get("title", btn.get("text", f"Button {i}"))}}
                    for i, btn in enumerate(buttons)
                ]
            }
        }
        
        if header_text:
            interactive["header"] = {"type": "text", "text": header_text}
        
        if footer_text:
            interactive["footer"] = {"text": footer_text}
        
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "interactive",
            "interactive": interactive
        }
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
    
    async def get_webhook_info(self) -> Dict[str, Any]:
        """
        Get webhook configuration info
        Note: WhatsApp doesn't provide an API endpoint to query webhook status.
        This returns the configured values from environment.
        """
        return {
            "platform": "whatsapp",
            "message": "WhatsApp webhook must be configured via Meta Business Suite",
            "phone_number_id": self.phone_number_id,
            "verify_token": os.getenv("WHATSAPP_VERIFY_TOKEN"),
            "webhook_configured": bool(self.access_token and self.phone_number_id),
            "instructions": "Check Meta Business Suite > WhatsApp > Configuration > Webhooks"
        }
