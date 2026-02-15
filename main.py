"""
Extended Brain - Main FastAPI Application
WhatsApp-powered knowledge base with Cerebras AI and PostgreSQL
Updated with full authentication and user registration
"""

# IMPORTANT: Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager
import os
from sqlalchemy.ext.asyncio import AsyncSession

# Import our modules
from database import get_db, init_db
from models import User, Message, Category, MessageType
from cerebras_client import CerebrasClient
from whatsapp import WhatsAppClient
from services.message_processor import MessageProcessor
from services.search_service import SearchService
from services.category_manager import CategoryManager
from services.auth_service import AuthService


# ================== Lifespan Events ==================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    print("🚀 Starting Extended Brain API...")
    await init_db()
    print("✓ Database initialized")
    print("✓ Extended Brain API started successfully")
    
    yield  # Application runs here
    
    # Shutdown
    print("✓ Extended Brain API shutdown")


# ================== FastAPI App ==================

app = FastAPI(
    title="Extended Brain API",
    description="WhatsApp-powered AI knowledge base with Cerebras and PostgreSQL",
    version="2.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-digital-mind.vercel.app",  # Add your Vercel domain
        "https://your-digital-mind-*.vercel.app",
        "http://localhost:5173",         # Keep for local dev
    ],  # Configure for production
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
cerebras_client = CerebrasClient(api_key=os.getenv("CEREBRAS_API_KEY"))
whatsapp_client = WhatsAppClient(
    access_token=os.getenv("WHATSAPP_ACCESS_TOKEN"),
    phone_number_id=os.getenv("WHATSAPP_PHONE_NUMBER_ID")
)
message_processor = MessageProcessor(cerebras_client)
search_service = SearchService(cerebras_client)
category_manager = CategoryManager(cerebras_client)
auth_service = AuthService(whatsapp_client)


# ================== Pydantic Models ==================

class MessageTypeEnum(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    VIDEO = "video"


class UserRegistrationRequest(BaseModel):
    """Full user registration data"""
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    age: int = Field(..., ge=13, le=120)
    occupation: str = Field(..., min_length=2, max_length=100)
    phone_number: str = Field(..., min_length=10, max_length=20)
    password: str = Field(..., min_length=6)
    timezone: Optional[str] = "UTC"


class OTPSendRequest(BaseModel):
    """Request to send OTP"""
    phone_number: str = Field(..., min_length=10, max_length=20)


class OTPVerifyRequest(BaseModel):
    """Request to verify OTP"""
    phone_number: str = Field(..., min_length=10, max_length=20)
    otp: str = Field(..., min_length=6, max_length=6)


class LoginRequest(BaseModel):
    """User login request"""
    phone_number: str
    password: str


class IncomingWebhook(BaseModel):
    """WhatsApp webhook payload"""
    object: str
    entry: List[Dict[str, Any]]


class MessageCreate(BaseModel):
    user_phone: str
    content: str
    message_type: MessageTypeEnum = MessageTypeEnum.TEXT
    media_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchQuery(BaseModel):
    user_phone: str
    query: str
    limit: int = 10
    category_filter: Optional[List[str]] = None


class CategoryOperation(BaseModel):
    user_phone: str
    operation: str
    category_name: Optional[str] = None
    new_name: Optional[str] = None
    description: Optional[str] = None


# ================== Core Endpoints ==================

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Extended Brain API",
        "version": "2.0.0",
        "status": "active",
        "features": [
            "WhatsApp Integration",
            "AI-Powered Categorization",
            "Semantic Search",
            "Multi-format Support (Text/Image/Audio/PDF)",
            "Dynamic Category Management",
            "OTP Authentication"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected",
            "cerebras": "active",
            "whatsapp": "active"
        }
    }


# ================== Authentication Endpoints ==================

@app.post("/api/auth/send-otp")
async def send_otp(
    request: OTPSendRequest,
    db: AsyncSession = Depends(get_db)
):
    """Send OTP to phone number"""
    result = await auth_service.send_otp(request.phone_number, db)
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['message'])
    
    return result


@app.post("/api/auth/verify-otp")
async def verify_otp(
    request: OTPVerifyRequest,
    db: AsyncSession = Depends(get_db)
):
    """Verify OTP code"""
    result = await auth_service.verify_otp(
        request.phone_number,
        request.otp,
        db
    )
    
    if not result['verified']:
        raise HTTPException(status_code=400, detail=result['message'])
    
    return result


@app.post("/api/auth/login")
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """User login"""
    result = await auth_service.login_user(
        request.phone_number,
        request.password,
        db
    )
    
    if not result['success']:
        raise HTTPException(status_code=401, detail=result['message'])
    
    return result


# ================== User Registration ==================

@app.post("/api/users/register")
async def register_user(
    user_data: UserRegistrationRequest,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user with full details"""
    
    result = await auth_service.register_user(
        phone_number=user_data.phone_number,
        name=user_data.name,
        email=user_data.email,
        age=user_data.age,
        occupation=user_data.occupation,
        password=user_data.password,
        timezone=user_data.timezone,
        db=db
    )
    
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['message'])
    
    return result


# ================== WhatsApp Webhook Endpoints ==================

@app.get("/webhook")
async def verify_webhook(
    hub_mode: str | None = None,
    hub_verify_token: str | None = None,
    hub_challenge: str | None = None
):
    """Verify WhatsApp webhook"""
    verify_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "your_verify_token")
    
    if hub_mode == "subscribe" and hub_verify_token == verify_token:
        return int(hub_challenge)
    
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def handle_webhook(
    webhook: IncomingWebhook,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Handle incoming WhatsApp messages"""
    
    background_tasks.add_task(
        process_whatsapp_message,
        webhook,
        db
    )
    
    return {"status": "received"}


async def process_whatsapp_message(webhook: IncomingWebhook, db: AsyncSession):
    """Process incoming WhatsApp message"""
    try:
        for entry in webhook.entry:
            for change in entry.get("changes", []):
                value = change.get("value", {})
                messages = value.get("messages", [])
                
                for message in messages:
                    phone = message.get("from")
                    message_type = message.get("type")
                    
                    # Extract content based on message type
                    content, media_url = await extract_message_content(message)
                    
                    # Process the message
                    result = await message_processor.process(
                        user_phone=phone,
                        content=content,
                        message_type=message_type,
                        media_url=media_url,
                        db=db
                    )
                    
                    # Handle different command types
                    if content.lower().startswith(("search:", "find:", "get:")):
                        search_result = await search_service.search(
                            user_phone=phone,
                            query=content.split(":", 1)[1].strip(),
                            db=db
                        )
                        response = format_search_results(search_result)
                    
                    elif content.lower().startswith(("category:", "categories")):
                        response = await handle_category_command(phone, content, db)
                    
                    else:
                        response = (
                            f"✓ Saved to '{result['category']}'!\n\n"
                            f"📊 Tags: {', '.join(result['tags'])}\n\n"
                            f"💡 Tip: Search with 'search: your query'"
                        )
                    
                    await whatsapp_client.send_message(phone, response)
                    
    except Exception as e:
        print(f"Error processing webhook: {e}")


async def extract_message_content(message: Dict) -> tuple[str, Optional[str]]:
    """Extract content and media URL from WhatsApp message"""
    message_type = message.get("type")
    
    if message_type == "text":
        return message.get("text", {}).get("body", ""), None
    
    elif message_type == "image":
        image_id = message.get("image", {}).get("id")
        caption = message.get("image", {}).get("caption", "")
        media_url = await whatsapp_client.get_media_url(image_id)
        return caption or "[Image]", media_url
    
    elif message_type == "audio":
        audio_id = message.get("audio", {}).get("id")
        media_url = await whatsapp_client.get_media_url(audio_id)
        transcription = await cerebras_client.transcribe_audio(media_url)
        return transcription, media_url
    
    elif message_type == "document":
        doc_id = message.get("document", {}).get("id")
        filename = message.get("document", {}).get("filename", "")
        media_url = await whatsapp_client.get_media_url(doc_id)
        from services.document_processor import extract_document_text
        text = await extract_document_text(media_url)
        return f"{filename}: {text}", media_url
    
    return "", None


# ================== Message Management Endpoints ==================

@app.post("/api/messages/capture")
async def capture_message(
    message: MessageCreate,
    db: AsyncSession = Depends(get_db)
):
    """Manually capture a message"""
    
    result = await message_processor.process(
        user_phone=message.user_phone,
        content=message.content,
        message_type=message.message_type,
        media_url=message.media_url,
        db=db
    )
    
    return {
        "success": True,
        "message": "Content captured successfully",
        "data": result
    }


@app.post("/api/search")
async def search_messages(
    search: SearchQuery,
    db: AsyncSession = Depends(get_db)
):
    """Search through user's knowledge base"""
    
    results = await search_service.search(
        user_phone=search.user_phone,
        query=search.query,
        limit=search.limit,
        category_filter=search.category_filter,
        db=db
    )
    
    return {
        "success": True,
        "query": search.query,
        "results": results,
        "total": len(results)
    }


# ================== Category Management ==================

@app.post("/api/categories/manage")
async def manage_categories(
    operation: CategoryOperation,
    db: AsyncSession = Depends(get_db)
):
    """Manage user categories"""
    
    if operation.operation == "list":
        categories = await category_manager.list_categories(operation.user_phone, db)
        return {"success": True, "categories": categories}
    
    elif operation.operation == "create":
        result = await category_manager.create_category(
            operation.user_phone,
            operation.category_name,
            operation.description,
            db
        )
        return {"success": True, "data": result}
    
    elif operation.operation == "edit":
        result = await category_manager.edit_category(
            operation.user_phone,
            operation.category_name,
            operation.new_name,
            operation.description,
            db
        )
        return {"success": True, "data": result}
    
    elif operation.operation == "delete":
        result = await category_manager.delete_category(
            operation.user_phone,
            operation.category_name,
            db
        )
        return {"success": True, "data": result}
    
    raise HTTPException(status_code=400, detail="Invalid operation")


async def handle_category_command(phone: str, content: str, db: AsyncSession) -> str:
    """Handle category-related commands"""
    content_lower = content.lower()
    
    if content_lower == "categories":
        categories = await category_manager.list_categories(phone, db)
        if not categories:
            return "You don't have any categories yet!"
        
        response = "📁 Your Categories:\n\n"
        for cat in categories:
            response += f"• {cat['name']} ({cat['count']} items)\n"
        return response
    
    return "Category command recognized. Use 'categories' to list all."


def format_search_results(results: List[Dict]) -> str:
    """Format search results for WhatsApp"""
    if not results:
        return "No results found."
    
    response = f"🔍 Found {len(results)} result(s):\n\n"
    
    for i, result in enumerate(results, 1):
        response += f"{i}. [{result['category']}] "
        response += f"{result['content'][:100]}...\n"
        response += f"   📅 {result['created_at']}\n\n"
    
    return response


# ================== Analytics ==================

@app.get("/api/analytics/{phone_number}")
async def get_user_analytics(
    phone_number: str,
    db: AsyncSession = Depends(get_db)
):
    """Get user analytics"""
    
    from sqlalchemy import select, func
    
    total_messages = await db.scalar(
        select(func.count(Message.id))
        .join(User)
        .where(User.phone_number == phone_number)
    )
    
    category_stats = await db.execute(
        select(Category.name, func.count(Message.id))
        .join(Message.category)
        .join(User)
        .where(User.phone_number == phone_number)
        .group_by(Category.name)
    )
    
    type_stats = await db.execute(
        select(Message.message_type, func.count(Message.id))
        .join(User)
        .where(User.phone_number == phone_number)
        .group_by(Message.message_type)
    )
    
    return {
        "total_messages": total_messages,
        "by_category": dict(category_stats.all()),
        "by_type": dict(type_stats.all()),
        "user": phone_number
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
