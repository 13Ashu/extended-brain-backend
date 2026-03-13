"""
Extended Brain - Main FastAPI Application
Multi-platform messaging support (WhatsApp/Telegram)
WhatsApp-powered knowledge base with Cerebras AI and PostgreSQL
Updated with platform abstraction and Telegram support
"""

# IMPORTANT: Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from contextlib import asynccontextmanager
import os
import asyncio
import httpx
from sqlalchemy.ext.asyncio import AsyncSession

# Import configuration and messaging
from config import Config, MessagingPlatform
from messaging_factory import create_messaging_client, get_platform_name
from messaging_interface import MessagingClient

# Import our modules
from database import get_db, init_db, engine, Base, async_session_maker
from models import User, Message, Category, MessageType
from cerebras_client import CerebrasClient
from services.message_processor import MessageProcessor
from services.search_service import SearchService
from services.category_manager import CategoryManager
from services.auth_service import AuthService, get_current_user
from services.reminder_service import ReminderService


# ================== Lifespan Events ==================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Extended Brain API...")

    await init_db()
    print("✓ Database initialized")

    # Create reminders table (safe — won't recreate if exists)
    from services.reminder_service import Reminder
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Start reminder scheduler
    scheduler_task = asyncio.create_task(
        reminder_service.run_scheduler(poll_interval=30)
    )
    print("✅ Reminder scheduler running")

    # Setup Telegram webhook
    if Config.get_messaging_platform() == MessagingPlatform.TELEGRAM and Config.TELEGRAM_WEBHOOK_URL:
        try:
            result = await messaging_client.setup_webhook(Config.TELEGRAM_WEBHOOK_URL)
            print(f"✓ Telegram webhook configured: {result}")
        except Exception as e:
            print(f"⚠ Telegram webhook setup failed: {e}")

    print("✓ Extended Brain API started successfully")
    yield

    # Cleanup on shutdown
    scheduler_task.cancel()
    print("✓ Extended Brain API shutdown")


# ================== FastAPI App ==================

app = FastAPI(
    title="Extended Brain API",
    description="Multi-platform AI knowledge base with Cerebras and PostgreSQL (WhatsApp/Telegram)",
    version="3.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080",
        "https://your-digital-mind.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services — ORDER MATTERS
cerebras_client = CerebrasClient()
messaging_client: MessagingClient = create_messaging_client()
search_service = SearchService(cerebras_client)
category_manager = CategoryManager(cerebras_client)
auth_service = AuthService(messaging_client)
reminder_service = ReminderService(cerebras_client)       # must be before message_processor
message_processor = MessageProcessor(cerebras_client, reminder_service=reminder_service)


# ================== Pydantic Models ==================

class MessageTypeEnum(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    VIDEO = "video"


class UserRegistrationRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: EmailStr
    age: int = Field(..., ge=13, le=120)
    occupation: str = Field(..., min_length=2, max_length=100)
    phone_number: str = Field(..., min_length=10, max_length=20)
    password: str = Field(..., min_length=6)
    timezone: Optional[str] = "Asia/Kolkata"


class OTPSendRequest(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=20)


class OTPVerifyRequest(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=20)
    otp: str = Field(..., min_length=6, max_length=6)


class LoginRequest(BaseModel):
    phone_number: str
    password: str


class ForgotPasswordRequest(BaseModel):
    phone_number: str
    new_password: str = Field(..., min_length=6)


class TelegramLinkRequest(BaseModel):
    phone_number: str
    telegram_chat_id: str


class MessageCreate(BaseModel):
    content: str
    message_type: MessageTypeEnum = MessageTypeEnum.TEXT
    media_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchQuery(BaseModel):
    query: str
    limit: int = 10
    category_filter: Optional[List[str]] = None


class CategoryOperation(BaseModel):
    operation: str
    category_name: Optional[str] = None
    new_name: Optional[str] = None
    description: Optional[str] = None


# ================== Core Endpoints ==================

@app.get("/")
async def root():
    return {
        "message": "Extended Brain API",
        "version": "3.0.0",
        "status": "active",
        "messaging_platform": get_platform_name(),
        "features": [
            f"{get_platform_name().capitalize()} Integration",
            "AI-Powered Categorization",
            "Semantic Search",
            "Multi-format Support (Text/Image/Audio/PDF)",
            "Dynamic Category Management",
            "OTP Authentication",
            "Reminders",
            "Interactive Todo Checklist",
        ]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "platform": get_platform_name(),
        "services": {
            "database": "connected",
            "cerebras": "active",
            "messaging": "active"
        }
    }


# ================== Authentication Endpoints ==================

@app.post("/api/auth/send-otp")
async def send_otp(request: OTPSendRequest, db: AsyncSession = Depends(get_db)):
    result = await auth_service.send_otp(request.phone_number, db)
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['message'])
    return result


@app.post("/api/auth/verify-otp")
async def verify_otp(request: OTPVerifyRequest, db: AsyncSession = Depends(get_db)):
    result = await auth_service.verify_otp(request.phone_number, request.otp, db)
    if not result['verified']:
        raise HTTPException(status_code=400, detail=result['message'])
    return result


@app.post("/api/auth/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await auth_service.login_user(request.phone_number, request.password, db)
    if not result['success']:
        raise HTTPException(status_code=401, detail=result['message'])
    return result


@app.post("/api/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    result = await auth_service.reset_password(
        phone_number=request.phone_number,
        new_password=request.new_password,
        db=db
    )
    if not result['success']:
        raise HTTPException(status_code=400, detail=result['message'])
    return result


@app.delete("/api/auth/delete-account")
async def delete_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    phone = current_user.phone_number
    await db.delete(current_user)
    await db.commit()
    print(f"🗑 Account deleted: {phone}")
    return {"success": True, "message": "Account and all associated data permanently deleted."}


@app.post("/api/auth/link-telegram")
async def link_telegram(request: TelegramLinkRequest, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    user = await db.scalar(select(User).where(User.phone_number == request.phone_number))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.telegram_chat_id = request.telegram_chat_id
    await db.commit()
    return {"success": True, "message": "Telegram linked successfully"}


# ================== User Registration ==================

@app.post("/api/users/register")
async def register_user(user_data: UserRegistrationRequest, db: AsyncSession = Depends(get_db)):
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


# ================== WhatsApp Webhook ==================

@app.get("/webhook/whatsapp")
async def verify_whatsapp_webhook(
    hub_mode: str | None = None,
    hub_verify_token: str | None = None,
    hub_challenge: str | None = None
):
    if Config.get_messaging_platform() != MessagingPlatform.WHATSAPP:
        raise HTTPException(status_code=400, detail="WhatsApp not configured")
    verify_token = Config.WHATSAPP_VERIFY_TOKEN
    if hub_mode == "subscribe" and hub_verify_token == verify_token:
        return int(hub_challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook/whatsapp")
async def handle_whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    if Config.get_messaging_platform() != MessagingPlatform.WHATSAPP:
        raise HTTPException(status_code=400, detail="WhatsApp not configured")
    webhook_data = await request.json()
    background_tasks.add_task(process_webhook_message, webhook_data, db)
    return {"status": "received"}


# ================== Telegram Webhook ==================

@app.post("/webhook/telegram")
async def handle_telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    if Config.get_messaging_platform() != MessagingPlatform.TELEGRAM:
        raise HTTPException(status_code=400, detail="Telegram not configured")

    webhook_data = await request.json()

    # Callback queries (button taps) need synchronous handling for fast ack
    if "callback_query" in webhook_data:
        await _handle_callback_query(webhook_data["callback_query"], db)
        return {"ok": True}

    background_tasks.add_task(process_webhook_message, webhook_data, db)
    return {"ok": True}


# ================== Telegram Callback Handler ==================

async def _handle_callback_query(callback: Dict, db: AsyncSession):
    """Handle inline keyboard button presses (todo checklist taps)"""
    callback_id  = callback["id"]
    chat_id      = str(callback["from"]["id"])
    callback_data = callback.get("data", "")
    tg_message_id = callback["message"]["message_id"]

    if not callback_data.startswith("done:"):
        await _answer_callback(callback_id, "Unknown action")
        return

    msg_db_id = int(callback_data.split(":")[1])

    # Mark message as done in DB
    async with async_session_maker() as session:
        from sqlalchemy import select, update
        msg = await session.scalar(select(Message).where(Message.id == msg_db_id))
        if msg:
            tags = dict(msg.tags or {})
            tags["done"] = True
            tags["done_at"] = datetime.utcnow().isoformat()
            await session.execute(
                update(Message).where(Message.id == msg_db_id).values(tags=tags)
            )
            await session.commit()

    # Acknowledge button press immediately
    await _answer_callback(callback_id, "✓ Done!")

    # Rebuild and update the checklist message
    await _refresh_checklist_message(chat_id, tg_message_id, db)


async def _answer_callback(callback_id: str, text: str = "✓ Done!"):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/answerCallbackQuery",
            json={"callback_query_id": callback_id, "text": text}
        )


async def _refresh_checklist_message(chat_id: str, tg_message_id: int, db: AsyncSession):
    from sqlalchemy import select

    async with async_session_maker() as session:  # use own session, not request-scoped db
        user = await session.scalar(select(User).where(User.telegram_chat_id == chat_id))
        if not user:
            return

    # Reuse search_service instead of raw SQL
    search_result = await search_service.search(
        user_phone=user.phone_number,
        query="todo tasks pending",
        db=db
    )
    items = []
    for r in search_result.get("results", []):
        tags = r.get("tags", {}) if isinstance(r.get("tags"), dict) else {}
        items.append({
            "id":         r["id"],
            "content":    r["content"],
            "event_time": r.get("event_time") or tags.get("event_time"),
            "tags":       tags,
        })

    items = [
        item for item in items
        if not item.get("tags", {}).get("done", False)
    ]

    text, reply_markup = format_todo_checklist(items)
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/editMessageText",
            json={
                "chat_id":      chat_id,
                "message_id":   tg_message_id,
                "text":         text,
                "parse_mode":   "Markdown",
                "reply_markup": reply_markup,
            }
        )


# ================== Checklist Helpers ==================
def format_todo_checklist(results: List[Dict]) -> tuple[str, dict]:
    """Returns (text, reply_markup) for Telegram inline checklist."""
    # Simple header only — no duplicate bullet list above the buttons
    text = "📋 *Your To-Do List*\n\n"
    buttons = []

    if not results:
        text += "_Nothing here yet!_"
        return text, {"inline_keyboard": buttons}

    for item in results:
        tags     = item.get("tags", {})
        is_done  = tags.get("done", False)
        due_time = item.get("event_time") or tags.get("event_time", "")
        time_str = f" {due_time}" if due_time else ""

        if is_done:
            # Strike-through done items in the header text only
            text += f"~✓ {item['content'][:40]}{time_str}~\n"
        else:
            # Each button is its own row — left aligned by default in Telegram
            buttons.append([{
                "text":          f"☐ {item['content'][:40]}{time_str}",
                "callback_data": f"done:{item['id']}"
            }])

    if all(item.get("tags", {}).get("done") for item in results):
        text += "_All done! 🎉_"

    return text, {"inline_keyboard": buttons}


async def send_todo_checklist(chat_id: str, results: List[Dict]):
    """Send todo results as an interactive checklist to Telegram."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    
    # Filter out already-done items
    results = [r for r in results if not r.get("tags", {}).get("done", False)]
    
    text, reply_markup = format_todo_checklist(results)
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id":      chat_id,
                "text":         text,
                "parse_mode":   "Markdown",
                "reply_markup": reply_markup,
            }
        )

# ================== Unified Message Processing ==================

TODO_SEARCH_KEYWORDS = {
    "todo", "to-do", "to do", "task", "tasks",
    "pending", "checklist", "check list"
}


async def process_webhook_message(webhook_data: Dict, db: AsyncSession):
    """Process incoming message from any platform"""
    try:
        from sqlalchemy import select

        messages = messaging_client.extract_message_data(webhook_data)

        for msg_data in messages:
            chat_id      = msg_data["user_id"]
            content      = msg_data["content"]
            message_type = msg_data["message_type"]
            metadata     = msg_data.get("metadata", {})

            # Look up user
            if Config.get_messaging_platform() == MessagingPlatform.TELEGRAM:
                result = await db.execute(select(User).where(User.telegram_chat_id == chat_id))
            else:
                result = await db.execute(select(User).where(User.phone_number == chat_id))
            user = result.scalar_one_or_none()

            # ── /start ────────────────────────────────────────────
            if content.lower().strip() == "/start":
                if user:
                    response = (
                        f"👋 Welcome back, {user.name}!\n\n"
                        f"Your Extended Brain is ready.\n\n"
                        f"📝 Send me anything to save it.\n\n"
                        f"🔍 Commands:\n"
                        f"• search: <query>\n"
                        f"• categories"
                    )
                else:
                    response = (
                        f"👋 Welcome to Extended Brain!\n\n"
                        f"1. Register at: https://your-digital-mind.vercel.app\n"
                        f"2. Link Telegram: /link +919876543210"
                    )
                await messaging_client.send_message(chat_id, response)
                continue

            # ── /link ─────────────────────────────────────────────
            if content.lower().startswith("/link"):
                parts = content.split()
                if len(parts) != 2:
                    await messaging_client.send_message(chat_id, "Usage: /link +919876543210")
                    continue
                phone = parts[1].strip()
                user_to_link = await db.scalar(select(User).where(User.phone_number == phone))
                if not user_to_link:
                    response = f"❌ No account found with {phone}\n\nPlease register first."
                else:
                    user_to_link.telegram_chat_id = chat_id
                    await db.commit()
                    response = f"✅ Linked! Hi {user_to_link.name}, you're all set."
                await messaging_client.send_message(chat_id, response)
                continue

            # ── Unregistered user ─────────────────────────────────
            if not user:
                await messaging_client.send_message(
                    chat_id,
                    "🚫 Please link your account first.\n\nSend: /link +919876543210"
                )
                continue

            user_phone = user.phone_number

            # ── Media handling ────────────────────────────────────
            media_url = None
            if metadata.get("media_id") or metadata.get("file_id"):
                media_id = metadata.get("media_id") or metadata.get("file_id")
                try:
                    media_url = await messaging_client.get_media_url(media_id)
                except Exception as e:
                    print(f"Error getting media URL: {e}")

            if message_type == "audio" and media_url:
                try:
                    content = await cerebras_client.transcribe_audio(media_url)
                except Exception as e:
                    print(f"Error transcribing audio: {e}")
                    content = "[Audio - transcription failed]"

            if message_type == "document" and media_url:
                try:
                    from services.document_processor import extract_document_text
                    extracted = await extract_document_text(media_url)
                    filename  = metadata.get("file_name", "document")
                    content   = f"{filename}: {extracted}"
                except Exception as e:
                    print(f"Error extracting document text: {e}")

            # ── Route message ─────────────────────────────────────
            try:
                # SEARCH
                if content.lower().startswith(("search:", "find:", "get:")):
                    query = content.split(":", 1)[1].strip()
                    search_result = await search_service.search(
                        user_phone=user_phone,
                        query=query,
                        db=db
                    )
                    results = search_result.get("results", [])

                    # Todo query → interactive checklist
                    is_todo_query = any(kw in query.lower() for kw in TODO_SEARCH_KEYWORDS)
                    if is_todo_query and results and Config.get_messaging_platform() == MessagingPlatform.TELEGRAM:
                        await send_todo_checklist(chat_id, results)
                        continue

                    response = format_search_results(search_result)

                # CATEGORIES
                elif content.lower().startswith(("category:", "categories")):
                    response = await handle_category_command(user_phone, content, db)

                # SAVE
                else:
                    result = await message_processor.process(
                        user_phone=user_phone,
                        content=content,
                        message_type=message_type,
                        media_url=media_url,
                        db=db
                    )
                    response = _build_save_response(result)

            except Exception as e:
                print(f"Processing error: {e}")
                import traceback
                traceback.print_exc()
                response = "⚠ Something went wrong. Please try again."

            await messaging_client.send_message(chat_id, response)

    except Exception as e:
        print(f"Error processing webhook: {e}")
        import traceback
        traceback.print_exc()


def _build_save_response(result: Dict) -> str:
    """Build a smart, context-aware save confirmation."""
    from zoneinfo import ZoneInfo

    category    = result.get("category", "Notes")
    all_buckets = result.get("all_buckets", [category])
    remind_at   = result.get("remind_at")
    due_date    = result.get("due_date")
    essence     = result.get("essence", "")
    split_count = result.get("split_count", 0)

    if remind_at:
        remind_dt = datetime.fromisoformat(remind_at).replace(
            tzinfo=ZoneInfo("UTC")
        ).astimezone(ZoneInfo("Asia/Kolkata"))
        time_str = remind_dt.strftime("%I:%M %p, %d %b")
        return (
            f"✓ Saved & reminder set!\n\n"
            f"⏰ {time_str} IST\n"
            f"📝 {essence}"
        )

    if split_count > 1:
        return (
            f"✓ Saved {split_count} tasks!\n\n"
            f"📝 {essence}"
        )

    if "To-Do" in all_buckets and due_date:
        return (
            f"✓ Added to To-Do!\n\n"
            f"📝 {essence}\n"
            f"📅 {due_date}"
        )

    return (
        f"✓ Saved to '{category}'!\n\n"
        f"📝 {essence}"
    )


# ================== Message Management Endpoints ==================

@app.post("/api/messages/capture")
async def capture_message(
    message: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await message_processor.process(
        user_phone=current_user.phone_number,
        content=message.content,
        message_type=message.message_type,
        media_url=message.media_url,
        db=db
    )
    return {"success": True, "message": "Content captured successfully", "data": result}


@app.post("/api/search")
async def search_messages(
    search: SearchQuery,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    search_data = await search_service.search(
        user_phone=current_user.phone_number,
        query=search.query,
        limit=search.limit,
        category_filter=search.category_filter,
        db=db
    )
    return {
        "success": True,
        "query":            search.query,
        "natural_response": search_data.get("natural_response", ""),
        "results":          search_data.get("results", []),
        "total":            len(search_data.get("results", []))
    }


# ================== Category Management ==================

@app.post("/api/categories/manage")
async def manage_categories(
    operation: CategoryOperation,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    phone = current_user.phone_number

    if operation.operation == "list":
        categories = await category_manager.list_categories(phone, db)
        return {"success": True, "categories": categories}
    elif operation.operation == "create":
        result = await category_manager.create_category(phone, operation.category_name, operation.description, db)
        return {"success": True, "data": result}
    elif operation.operation == "edit":
        result = await category_manager.edit_category(phone, operation.category_name, operation.new_name, operation.description, db)
        return {"success": True, "data": result}
    elif operation.operation == "delete":
        result = await category_manager.delete_category(phone, operation.category_name, db)
        return {"success": True, "data": result}

    raise HTTPException(status_code=400, detail="Invalid operation")


async def handle_category_command(phone: str, content: str, db: AsyncSession) -> str:
    if content.lower() == "categories":
        categories = await category_manager.list_categories(phone, db)
        if not categories:
            return "You don't have any categories yet!"
        response = "📁 *Your Categories*\n\n"
        for cat in categories:
            response += f"• {cat['name']} ({cat['count']} items)\n"
        return response
    return "Send 'categories' to list all your categories."


def format_search_results(search_data) -> str:
    if isinstance(search_data, list):
        return "No results found." if not search_data else f"Found {len(search_data)} result(s)."
    natural = search_data.get("natural_response", "")
    results = search_data.get("results", [])
    if not results:
        return "I couldn't find anything related to that in your notes."
    # Guard against fallback string from failed LLM call
    if not natural or natural.strip() in ("{}", ""):
        return "\n".join(f"• {r['content']}" for r in results[:5])
    return natural


# ================== Analytics ==================

@app.get("/api/analytics")
async def get_user_analytics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    from sqlalchemy import select, func

    phone_number = current_user.phone_number
    total_messages = await db.scalar(
        select(func.count(Message.id)).join(User).where(User.phone_number == phone_number)
    )
    category_stats = await db.execute(
        select(Category.name, func.count(Message.id))
        .join(Message.category).join(User)
        .where(User.phone_number == phone_number)
        .group_by(Category.name)
    )
    type_stats = await db.execute(
        select(Message.message_type, func.count(Message.id))
        .join(User).where(User.phone_number == phone_number)
        .group_by(Message.message_type)
    )
    return {
        "total_messages": total_messages,
        "by_category":    dict(category_stats.all()),
        "by_type":        dict(type_stats.all()),
        "user":           phone_number
    }


# ================== Webhook Info ==================

@app.get("/api/webhook/info")
async def get_webhook_info():
    platform = Config.get_messaging_platform()
    if platform == MessagingPlatform.WHATSAPP:
        return {
            "platform":         "whatsapp",
            "webhook_endpoint": "/webhook/whatsapp",
            "verify_token":     Config.WHATSAPP_VERIFY_TOKEN,
        }
    elif platform == MessagingPlatform.TELEGRAM:
        try:
            info = await messaging_client.get_webhook_info()
            return {
                "platform":         "telegram",
                "webhook_endpoint": "/webhook/telegram",
                "current_webhook":  info.get("result", {}),
                "configured_url":   Config.TELEGRAM_WEBHOOK_URL
            }
        except Exception:
            return {
                "platform":         "telegram",
                "webhook_endpoint": "/webhook/telegram",
                "configured_url":   Config.TELEGRAM_WEBHOOK_URL,
                "status":           "not_configured"
            }