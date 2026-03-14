"""
Extended Brain - Main FastAPI Application
Multi-platform messaging support (WhatsApp/Telegram)
"""

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

from config import Config, MessagingPlatform
from messaging_factory import create_messaging_client, get_platform_name
from messaging_interface import MessagingClient

from database import get_db, init_db, engine, Base, async_session_maker
from models import User, Message, Category, MessageType
from cerebras_client import CerebrasClient
from services.message_processor import MessageProcessor
from services.search_service import SearchService
from services.category_manager import CategoryManager
from services.auth_service import AuthService, get_current_user
from services.reminder_service import ReminderService
from services.briefing_service import briefing_service
from services.nudge_service import nudge_service
from services.context_service import context_service
from services.project_service import ProjectService
from services.subtask_service import SubtaskService
from services.recurrence_service import RecurrenceService, Recurrence


# ================== Lifespan ==================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Extended Brain API...")
    await init_db()
    print("✓ Database initialized")

    # Create all new tables
    from services.reminder_service import Reminder
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Start scheduler
    scheduler_task = asyncio.create_task(_master_scheduler())
    print("✅ Master scheduler running")

    if Config.get_messaging_platform() == MessagingPlatform.TELEGRAM and Config.TELEGRAM_WEBHOOK_URL:
        try:
            result = await messaging_client.setup_webhook(Config.TELEGRAM_WEBHOOK_URL)
            print(f"✓ Telegram webhook configured: {result}")
        except Exception as e:
            print(f"⚠ Telegram webhook setup failed: {e}")

    print("✓ Extended Brain API started successfully")
    yield
    scheduler_task.cancel()
    print("✓ Extended Brain API shutdown")


async def _master_scheduler():
    """
    Single scheduler loop — runs every 60 seconds.
    Delegates to each sub-scheduler.
    """
    from services.reminder_service import ReminderService as RS
    print("[scheduler] Master scheduler started")
    tick = 0
    while True:
        try:
            await asyncio.sleep(60)
            tick += 1

            # Every minute: reminders + recurrences + briefing check
            await reminder_service.run_scheduler_tick()
            await recurrence_service.run()
            await briefing_service.run()

            # Every 30 minutes: nudges + follow-ups
            if tick % 30 == 0:
                await nudge_service.run_idle_nudges()
                await nudge_service.run_followup_checks()

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[scheduler] Error: {e}")


# ================== FastAPI App ==================

app = FastAPI(
    title="Extended Brain API",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080",
        "https://your-digital-mind.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services — ORDER MATTERS
cerebras_client    = CerebrasClient()
messaging_client: MessagingClient = create_messaging_client()
search_service     = SearchService(cerebras_client)
category_manager   = CategoryManager(cerebras_client)
auth_service       = AuthService(messaging_client)
reminder_service   = ReminderService(cerebras_client)
message_processor  = MessageProcessor(cerebras_client, reminder_service=reminder_service)
project_service    = ProjectService(cerebras_client)
subtask_service    = SubtaskService(cerebras_client)
recurrence_service = RecurrenceService(cerebras_client)


# ================== Pydantic Models ==================

class MessageTypeEnum(str, Enum):
    TEXT     = "text"
    IMAGE    = "image"
    AUDIO    = "audio"
    DOCUMENT = "document"
    VIDEO    = "video"


class UserRegistrationRequest(BaseModel):
    name:         str      = Field(..., min_length=2, max_length=100)
    email:        EmailStr
    age:          int      = Field(..., ge=13, le=120)
    occupation:   str      = Field(..., min_length=2, max_length=100)
    phone_number: str      = Field(..., min_length=10, max_length=20)
    password:     str      = Field(..., min_length=6)
    timezone:     Optional[str] = "Asia/Kolkata"


class OTPSendRequest(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=20)


class OTPVerifyRequest(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=20)
    otp:          str = Field(..., min_length=6, max_length=6)


class LoginRequest(BaseModel):
    phone_number: str
    password:     str


class ForgotPasswordRequest(BaseModel):
    phone_number: str
    new_password: str = Field(..., min_length=6)


class TelegramLinkRequest(BaseModel):
    phone_number:    str
    telegram_chat_id: str


class MessageCreate(BaseModel):
    content:      str
    message_type: MessageTypeEnum = MessageTypeEnum.TEXT
    media_url:    Optional[str]   = None
    metadata:     Optional[Dict[str, Any]] = None


class SearchQuery(BaseModel):
    query:           str
    limit:           int = 10
    category_filter: Optional[List[str]] = None


class CategoryOperation(BaseModel):
    operation:    str
    category_name: Optional[str] = None
    new_name:     Optional[str]  = None
    description:  Optional[str]  = None


# ================== Core Endpoints ==================

@app.get("/")
async def root():
    return {
        "message": "Extended Brain API",
        "version": "4.0.0",
        "status":  "active",
        "features": [
            "Morning Briefing", "Recurring Tasks", "Subtasks",
            "Project Grouping", "Idle Nudges", "Follow-up Tracking",
            "Multi-turn Context", "Priority Escalation", "/status",
        ],
    }


@app.get("/health")
async def health_check():
    return {
        "status":    "healthy",
        "timestamp": datetime.now().isoformat(),
        "platform":  get_platform_name(),
    }


# ================== Auth Endpoints ==================

@app.post("/api/auth/send-otp")
async def send_otp(request: OTPSendRequest, db: AsyncSession = Depends(get_db)):
    result = await auth_service.send_otp(request.phone_number, db)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/auth/verify-otp")
async def verify_otp(request: OTPVerifyRequest, db: AsyncSession = Depends(get_db)):
    result = await auth_service.verify_otp(request.phone_number, request.otp, db)
    if not result["verified"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/auth/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await auth_service.login_user(request.phone_number, request.password, db)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["message"])
    return result


@app.post("/api/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    result = await auth_service.reset_password(
        phone_number=request.phone_number, new_password=request.new_password, db=db
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.delete("/api/auth/delete-account")
async def delete_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    phone = current_user.phone_number
    await db.delete(current_user)
    await db.commit()
    return {"success": True, "message": "Account deleted."}


@app.post("/api/auth/link-telegram")
async def link_telegram(request: TelegramLinkRequest, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    user = await db.scalar(select(User).where(User.phone_number == request.phone_number))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.telegram_chat_id = request.telegram_chat_id
    await db.commit()
    return {"success": True, "message": "Telegram linked successfully"}


@app.post("/api/users/register")
async def register_user(user_data: UserRegistrationRequest, db: AsyncSession = Depends(get_db)):
    result = await auth_service.register_user(
        phone_number=user_data.phone_number, name=user_data.name,
        email=user_data.email, age=user_data.age, occupation=user_data.occupation,
        password=user_data.password, timezone=user_data.timezone, db=db,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


# ================== WhatsApp Webhook ==================

@app.get("/webhook/whatsapp")
async def verify_whatsapp_webhook(
    hub_mode: str | None = None,
    hub_verify_token: str | None = None,
    hub_challenge: str | None = None,
):
    if Config.get_messaging_platform() != MessagingPlatform.WHATSAPP:
        raise HTTPException(status_code=400, detail="WhatsApp not configured")
    if hub_mode == "subscribe" and hub_verify_token == Config.WHATSAPP_VERIFY_TOKEN:
        return int(hub_challenge)
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook/whatsapp")
async def handle_whatsapp_webhook(
    request: Request, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
):
    if Config.get_messaging_platform() != MessagingPlatform.WHATSAPP:
        raise HTTPException(status_code=400, detail="WhatsApp not configured")
    webhook_data = await request.json()
    background_tasks.add_task(process_webhook_message, webhook_data, db)
    return {"status": "received"}


# ================== Telegram Webhook ==================

@app.post("/webhook/telegram")
async def handle_telegram_webhook(
    request: Request, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)
):
    if Config.get_messaging_platform() != MessagingPlatform.TELEGRAM:
        raise HTTPException(status_code=400, detail="Telegram not configured")

    webhook_data = await request.json()

    if "callback_query" in webhook_data:
        await _handle_callback_query(webhook_data["callback_query"], db)
        return {"ok": True}

    background_tasks.add_task(process_webhook_message, webhook_data, db)
    return {"ok": True}


# ================== Callback Handler ==================

async def _handle_callback_query(callback: Dict, db: AsyncSession):
    callback_id   = callback["id"]
    chat_id       = str(callback["from"]["id"])
    callback_data = callback.get("data", "")
    tg_message_id = callback["message"]["message_id"]

    # ── done:<id> ─────────────────────────────────────────────────
    if callback_data.startswith("done:"):
        msg_db_id = int(callback_data.split(":")[1])
        async with async_session_maker() as session:
            from sqlalchemy import select, update
            msg = await session.scalar(select(Message).where(Message.id == msg_db_id))
            if msg:
                tags = dict(msg.tags or {})
                tags["done"]    = True
                tags["done_at"] = datetime.utcnow().isoformat()
                await session.execute(
                    update(Message).where(Message.id == msg_db_id).values(tags=tags)
                )
                await session.commit()
        await _answer_callback(callback_id, "✓ Done!")
        await _refresh_checklist_message(chat_id, tg_message_id)

    # ── snooze:<id>:<minutes> ─────────────────────────────────────
    elif callback_data.startswith("snooze:"):
        parts     = callback_data.split(":")
        msg_id    = int(parts[1])
        minutes   = int(parts[2]) if len(parts) > 2 else 1440
        await nudge_service.snooze_message(msg_id, minutes)
        await _answer_callback(callback_id, f"⏰ Snoozed for {minutes // 60}h")

    # ── subtask:<message_id>:<index> ──────────────────────────────
    elif callback_data.startswith("subtask:"):
        parts   = callback_data.split(":")
        msg_id  = int(parts[1])
        idx     = int(parts[2])
        await subtask_service.complete_subtask(msg_id, idx)
        await _answer_callback(callback_id, "✓ Subtask done!")
        # Refresh subtask view
        await _refresh_subtask_message(chat_id, tg_message_id, msg_id)

    # ── pause_rec:<rec_id> ────────────────────────────────────────
    elif callback_data.startswith("pause_rec:"):
        rec_id = int(callback_data.split(":")[1])
        await recurrence_service.pause(rec_id)
        await _answer_callback(callback_id, "⏸ Recurring task paused")

    # ── set_briefing_time ─────────────────────────────────────────
    elif callback_data == "set_briefing_time":
        await _answer_callback(callback_id, "")
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id":    chat_id,
                    "text":       (
                        "⏰ *Set your morning briefing time*\n\n"
                        "Reply with the time in IST, e.g.:\n"
                        "`briefing: 07:30`\n`briefing: 08:00`"
                    ),
                    "parse_mode": "Markdown",
                },
            )

    # ── project confirm: yes:<msg_id>:<project> ───────────────────
    elif callback_data.startswith("proj_yes:"):
        parts      = callback_data.split(":", 2)
        msg_id     = int(parts[1])
        proj_name  = parts[2]
        await project_service.assign_project(msg_id, proj_name)
        await _answer_callback(callback_id, f"📁 Added to {proj_name}")
        await context_service.clear_pending_confirmation(int(chat_id))

    elif callback_data.startswith("proj_no:"):
        await _answer_callback(callback_id, "OK, not grouped")
        await context_service.clear_pending_confirmation(int(chat_id))

    else:
        await _answer_callback(callback_id, "Unknown action")


async def _answer_callback(callback_id: str, text: str = ""):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/answerCallbackQuery",
            json={"callback_query_id": callback_id, "text": text},
        )


# ================== Checklist & Subtask Refresh ==================

async def _refresh_checklist_message(chat_id: str, tg_message_id: int):
    async with async_session_maker() as session:
        from sqlalchemy import select
        user = await session.scalar(select(User).where(User.telegram_chat_id == chat_id))
        if not user:
            return
        search_result = await search_service.search(
            user_phone=user.phone_number, query="todo tasks pending", db=session
        )

    items = [
        {
            "id":         r["id"],
            "content":    r["content"],
            "event_time": r.get("event_time") or r.get("tags", {}).get("event_time"),
            "tags":       r.get("tags", {}),
        }
        for r in search_result.get("results", [])
        if not r.get("tags", {}).get("done", False)
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
            },
        )


async def _refresh_subtask_message(chat_id: str, tg_message_id: int, message_id: int):
    async with async_session_maker() as session:
        from sqlalchemy import select
        msg = await session.scalar(select(Message).where(Message.id == message_id))
        if not msg:
            return

    text, reply_markup = subtask_service.format_subtasks(msg)
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
            },
        )


# ================== Checklist Helpers ==================

def format_todo_checklist(results: List[Dict]) -> tuple[str, dict]:
    text    = "📋 *Your To-Do List*\n\n"
    buttons = []

    if not results:
        text += "_Nothing here yet!_"
        return text, {"inline_keyboard": buttons}

    for item in results:
        tags     = item.get("tags", {})
        is_done  = tags.get("done", False)
        due_time = item.get("event_time") or tags.get("event_time", "")
        time_str = f" {due_time}" if due_time else ""
        has_subs = bool(tags.get("subtasks"))
        sub_icon = " 📎" if has_subs else ""

        if is_done:
            text += f"~✓ {item['content'][:40]}{time_str}~\n"
        else:
            buttons.append([{
                "text":          f"☐ {item['content'][:38]}{time_str}{sub_icon}",
                "callback_data": f"done:{item['id']}",
            }])

    if results and all(item.get("tags", {}).get("done") for item in results):
        text += "_All done! 🎉_"

    return text, {"inline_keyboard": buttons}


async def send_todo_checklist(chat_id: str, results: List[Dict], date_from: str = None, date_to: str = None):
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
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
            },
        )


# ================== Message Processing ==================

TODO_SEARCH_KEYWORDS = {
    "todo", "to-do", "to do", "task", "tasks",
    "pending", "checklist", "check list",
}


def _is_search_query(content: str) -> bool:
    """True if content starts with search:/find:/get: OR ends with ?"""
    lc = content.lower().strip()
    return (
        lc.startswith(("search:", "find:", "get:"))
        or lc.endswith("?")
    )


def _extract_query(content: str) -> str:
    """Extract the actual query string."""
    lc = content.lower().strip()
    if lc.startswith(("search:", "find:", "get:")):
        return content.split(":", 1)[1].strip()
    # Ends with ?
    return content.strip()


async def process_webhook_message(webhook_data: Dict, db: AsyncSession):
    try:
        from sqlalchemy import select, update

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
                        f"📝 Send anything to save it\n"
                        f"🔍 *Search:* `search: query` or end with ?\n"
                        f"✅ *Todo:* `search: my todos`\n"
                        f"📊 *Status:* `/status`\n"
                        f"📁 *Projects:* `project: Japan trip`\n"
                        f"🔁 *Recurring:* `every monday: weekly review`\n"
                        f"📎 *Subtasks:* `subtask: parent > task1, task2`\n"
                        f"⏰ *Briefing:* `briefing: 08:00`"
                    )
                else:
                    response = (
                        f"👋 Welcome to Extended Brain!\n\n"
                        f"1. Register at: https://your-digital-mind.vercel.app\n"
                        f"2. Link Telegram: `/link +919876543210`"
                    )
                await messaging_client.send_message(chat_id, response)
                continue

            # ── /link ─────────────────────────────────────────────
            if content.lower().startswith("/link"):
                parts = content.split()
                if len(parts) != 2:
                    await messaging_client.send_message(chat_id, "Usage: /link +919876543210")
                    continue
                phone        = parts[1].strip()
                user_to_link = await db.scalar(select(User).where(User.phone_number == phone))
                if not user_to_link:
                    response = f"❌ No account found with {phone}"
                else:
                    user_to_link.telegram_chat_id = chat_id
                    await db.commit()
                    response = f"✅ Linked! Hi {user_to_link.name}, you're all set."
                await messaging_client.send_message(chat_id, response)
                continue

            if not user:
                await messaging_client.send_message(
                    chat_id,
                    "🚫 Please link your account first.\n\nSend: /link +919876543210"
                )
                continue

            user_phone = user.phone_number

            # ── /status ───────────────────────────────────────────
            if content.lower().strip() in {"/status", "status"}:
                response = await _build_status(user, db)
                await messaging_client.send_message(chat_id, response)
                continue

            # ── briefing time setter ───────────────────────────────
            if content.lower().startswith("briefing:"):
                time_str = content.split(":", 1)[1].strip()
                import re
                if re.match(r"^\d{1,2}:\d{2}$", time_str):
                    # Normalise to HH:MM
                    h, m     = time_str.split(":")
                    hhmm     = f"{int(h):02d}:{int(m):02d}"
                    ok       = await briefing_service.set_briefing_time(user.id, hhmm)
                    response = (
                        f"✅ Morning briefing set to *{hhmm} IST* daily!"
                        if ok else "❌ Failed to update briefing time."
                    )
                else:
                    response = "❌ Use format `briefing: 08:00`"
                await messaging_client.send_message(chat_id, response)
                continue

            # ── project view command ───────────────────────────────
            if content.lower().startswith("project:"):
                project_name = content.split(":", 1)[1].strip()
                msgs         = await project_service.get_project_messages(user.id, project_name, db)
                response     = project_service.format_project_summary(project_name, msgs)
                await messaging_client.send_message(chat_id, response)
                continue

            # ── pending confirmation (project/subtask) ────────────
            pending = await context_service.get_pending_confirmation(user.id)
            if pending:
                lc = content.lower().strip()
                if lc in {"yes", "y", "yeah", "yep", "sure", "ok", "okay"}:
                    await _handle_confirmation_yes(user, pending, chat_id, db)
                    await context_service.clear_pending_confirmation(user.id)
                    continue
                elif lc in {"no", "n", "nope", "nah", "skip"}:
                    await context_service.clear_pending_confirmation(user.id)
                    await messaging_client.send_message(chat_id, "👍 Skipped.")
                    continue

            # ── Media handling ─────────────────────────────────────
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
                except Exception:
                    content = "[Audio - transcription failed]"

            if message_type == "document" and media_url:
                try:
                    from services.document_processor import extract_document_text
                    extracted = await extract_document_text(media_url)
                    content   = f"{metadata.get('file_name', 'document')}: {extracted}"
                except Exception as e:
                    print(f"Error extracting document: {e}")

            try:
                # ── SEARCH (search: prefix OR ends with ?) ─────────
                if _is_search_query(content):
                    query         = _extract_query(content)
                    prev_ctx      = await context_service.get_search_context(user.id)

                    # If this is a follow-up, prepend previous context
                    if prev_ctx and not content.lower().startswith(("search:", "find:", "get:")):
                        # Ends with ? and there's a previous search — it's a follow-up
                        query = f"{prev_ctx['query']} {query}"

                    search_result = await search_service.search(
                        user_phone=user_phone, query=query, db=db
                    )
                    results       = search_result.get("results", [])

                    # Store context for follow-up
                    natural = search_result.get("natural_response", "")
                    await context_service.set_search_context(user.id, query, natural[:300])

                    # Named list query → list checklist display (zero LLM render)
                    if search_result.get("is_list") and Config.get_messaging_platform() == MessagingPlatform.TELEGRAM:
                        await send_list_display(chat_id, search_result)
                        continue

                    # Todo query → todo checklist display
                    is_todo_query = any(kw in query.lower() for kw in TODO_SEARCH_KEYWORDS)
                    if is_todo_query and Config.get_messaging_platform() == MessagingPlatform.TELEGRAM:
                        date_from = search_result.get("date_from")
                        date_to   = search_result.get("date_to")
                        await send_todo_checklist(chat_id, results, date_from, date_to)
                        continue

                    response = format_search_results(search_result)

                # ── CATEGORIES ─────────────────────────────────────
                elif content.lower().startswith(("category:", "categories")):
                    response = await handle_category_command(user_phone, content, db)

                # ── SUBTASK COMMAND ────────────────────────────────
                elif content.lower().startswith("subtask:"):
                    response = await _handle_subtask_command(user, content, chat_id, db)

                # ── RECURRING TASK ─────────────────────────────────
                elif recurrence_service.is_recurring(content):
                    response = await _handle_recurring(user, content, db)

                # ── SAVE ───────────────────────────────────────────
                else:
                    # Check for NL subtask intent first
                    subtask_intent = await subtask_service.detect_subtask_intent(
                        content, user.id, db
                    )
                    if subtask_intent:
                        response = await _handle_nl_subtask(user, subtask_intent, content, chat_id, db)
                    else:
                        result   = await message_processor.process(
                            user_phone=user_phone, content=content,
                            message_type=message_type, media_url=media_url, db=db,
                        )
                        response = _build_save_response(result)

                        # Project detection (async, non-blocking for response)
                        asyncio.create_task(
                            _check_project(user, result, content, chat_id, db)
                        )

                        # Clear search context since user saved something new
                        await context_service.clear_search_context(user.id)

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


# ================== Feature Handlers ==================

async def _build_status(user: User, db: AsyncSession) -> str:
    from sqlalchemy import select, func, text
    from datetime import date, timedelta

    today      = datetime.utcnow().strftime("%Y-%m-%d")
    week_start = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

    # Pending todos today
    pending_res = await db.execute(
        select(func.count(Message.id))
        .where(
            Message.user_id == user.id,
            text("messages.tags->>'due_date' = :today"),
            text("(messages.tags->>'done')::boolean IS NOT TRUE"),
            text("messages.tags->'all_buckets' @> '\"To-Do\"'::jsonb"),
        )
        .params(today=today)
    )
    pending = pending_res.scalar() or 0

    # Done today
    done_res = await db.execute(
        select(func.count(Message.id))
        .where(
            Message.user_id == user.id,
            text("(messages.tags->>'done')::boolean IS TRUE"),
            text("messages.tags->>'done_at' >= :today"),
        )
        .params(today=f"{today}T00:00:00")
    )
    done_today = done_res.scalar() or 0

    # Saves this week
    saves_res = await db.execute(
        select(func.count(Message.id))
        .where(
            Message.user_id == user.id,
            Message.created_at >= week_start,
        )
    )
    saves_week = saves_res.scalar() or 0

    # Ideas count
    ideas_res = await db.execute(
        select(func.count(Message.id))
        .where(
            Message.user_id == user.id,
            text("messages.tags->'all_buckets' @> '\"Ideas\"'::jsonb"),
        )
    )
    ideas = ideas_res.scalar() or 0

    # Events this week
    week_end = (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d")
    events_res = await db.execute(
        select(func.count(Message.id))
        .where(
            Message.user_id == user.id,
            text("messages.tags->>'due_date' BETWEEN :start AND :end"),
            text("messages.tags->'all_buckets' @> '\"Events\"'::jsonb"),
        )
        .params(start=today, end=week_end)
    )
    events = events_res.scalar() or 0

    # Active recurrences
    rec_res = await db.execute(
        select(func.count())
        .select_from(text("recurrences"))
        .where(
            text("user_id = :uid AND is_active = TRUE")
        )
        .params(uid=user.id)
    )
    recurrences = rec_res.scalar() or 0

    briefing_time = getattr(user, "briefing_time", "08:00") or "08:00"

    return (
        f"📊 *Your Brain — {user.name}*\n\n"
        f"✅ Done today: *{done_today}*\n"
        f"⏳ Pending today: *{pending}*\n"
        f"📅 Events this week: *{events}*\n"
        f"💡 Total ideas: *{ideas}*\n"
        f"📝 Saves this week: *{saves_week}*\n"
        f"🔁 Active recurring tasks: *{recurrences}*\n"
        f"⏰ Morning briefing: *{briefing_time} IST*\n\n"
        f"_Reply 'briefing: HH:MM' to change briefing time_"
    )


async def _handle_subtask_command(
    user: User, content: str, chat_id: str, db: AsyncSession
) -> str:
    parsed = subtask_service.parse_command(content)
    if not parsed:
        return (
            "❌ Subtask format:\n"
            "`subtask: <parent task> > <sub1>, <sub2>`\n\n"
            "Example:\n`subtask: Japan trip > book flights, get visa, pack bags`"
        )

    parent_query, subs = parsed
    candidates         = await subtask_service.find_parent_message(user.id, parent_query, db)

    if not candidates:
        return f"❌ Couldn't find a task matching *{parent_query}*. Try being more specific."

    if len(candidates) == 1:
        ok = await subtask_service.add_subtasks(candidates[0].id, subs)
        if ok:
            text_, reply_markup = subtask_service.format_subtasks(candidates[0])
            # Re-fetch to get updated tags
            async with async_session_maker() as session:
                from sqlalchemy import select as sel
                msg = await session.scalar(sel(Message).where(Message.id == candidates[0].id))
            text_, reply_markup = subtask_service.format_subtasks(msg)
            token = os.getenv("TELEGRAM_BOT_TOKEN", "")
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(
                    f"https://api.telegram.org/bot{token}/sendMessage",
                    json={
                        "chat_id":      chat_id,
                        "text":         text_,
                        "parse_mode":   "Markdown",
                        "reply_markup": reply_markup,
                    },
                )
            return ""  # Already sent above
        return "❌ Failed to add subtasks."

    # Multiple candidates — ask user to pick
    options = "\n".join(
        f"{i+1}. {c.content[:60]}" for i, c in enumerate(candidates[:5])
    )
    # Store pending confirmation with candidates
    await context_service.set_pending_confirmation(user.id, "subtask_pick", {
        "candidates": [{"id": c.id, "content": c.content} for c in candidates[:5]],
        "subtasks":   subs,
    })
    return (
        f"🤔 Found {len(candidates)} matching tasks:\n\n{options}\n\n"
        f"Which one? Reply with the number (1-{min(len(candidates),5)})"
    )


async def _handle_nl_subtask(
    user: User, intent: Dict, content: str, chat_id: str, db: AsyncSession
) -> str:
    parent_id = intent.get("parent_id")
    subs      = intent.get("subtasks", [])
    hint      = intent.get("parent_hint", "")

    if parent_id:
        # High confidence match
        await context_service.set_pending_confirmation(user.id, "subtask_confirm", {
            "message_id": parent_id,
            "subtasks":   subs,
            "hint":       hint,
        })
        return (
            f"📎 Add these subtasks to *{hint}*?\n\n"
            + "\n".join(f"• {s}" for s in subs)
            + "\n\nReply *yes* to confirm or *no* to skip."
        )
    else:
        # Ambiguous — search for parent
        candidates = await subtask_service.find_parent_message(user.id, hint, db)
        if not candidates:
            return f"❌ Couldn't find a task matching *{hint}*."
        await context_service.set_pending_confirmation(user.id, "subtask_pick", {
            "candidates": [{"id": c.id, "content": c.content} for c in candidates[:5]],
            "subtasks":   subs,
        })
        options = "\n".join(
            f"{i+1}. {c.content[:60]}" for i, c in enumerate(candidates[:5])
        )
        return (
            f"🤔 Which task should these subtasks go under?\n\n{options}\n\n"
            f"Reply with the number."
        )


async def _handle_recurring(user: User, content: str, db: AsyncSession) -> str:
    parsed = await recurrence_service.parse_recurrence(content)
    if not parsed:
        return (
            "❌ Couldn't parse the recurring task.\n\n"
            "Try: `every monday: weekly review` or `drink water every day at 8am`"
        )

    rec = await recurrence_service.create(user.id, parsed, content, db)
    if not rec:
        return "❌ Failed to create recurring task."

    rule_str = recurrence_service._rule_display(rec)
    return (
        f"🔁 *Recurring task created!*\n\n"
        f"📝 {parsed['task']}\n"
        f"⏰ {rule_str}\n\n"
        f"_I'll remind you automatically._"
    )


async def _check_project(
    user: User, result: Dict, content: str, chat_id: str, db: AsyncSession
):
    """Background task — detect project and ask user if found."""
    try:
        async with async_session_maker() as session:
            msg = await session.scalar(
                __import__("sqlalchemy").select(Message).where(
                    Message.id == result.get("message_id")
                )
            )
            if not msg:
                return

            # Build a minimal analysis dict for project detection
            analysis = {
                "concepts": result.get("connections", []),
                "keywords": result.get("tags", []),
            }

            project = await project_service.detect_and_suggest(user, msg, analysis, session)
            if not project:
                return

        # Store pending confirmation
        await context_service.set_pending_confirmation(user.id, "project_confirm", {
            "message_id": msg.id,
            "project":    project,
        })

        # Send inline confirmation
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id":    chat_id,
                    "text":       f"📁 Add this to project *{project}*?",
                    "parse_mode": "Markdown",
                    "reply_markup": {
                        "inline_keyboard": [[
                            {"text": "✅ Yes",  "callback_data": f"proj_yes:{msg.id}:{project}"},
                            {"text": "❌ No",   "callback_data": f"proj_no:{msg.id}"},
                        ]]
                    },
                },
            )
    except Exception as e:
        print(f"[project] Background check failed: {e}")


async def _handle_confirmation_yes(
    user: User, pending: Dict, chat_id: str, db: AsyncSession
):
    conf_type = pending.get("type")
    data      = pending.get("data", {})

    if conf_type == "subtask_confirm":
        ok = await subtask_service.add_subtasks(data["message_id"], data["subtasks"])
        msg_text = "✅ Subtasks added!" if ok else "❌ Failed to add subtasks."
        await messaging_client.send_message(chat_id, msg_text)

    elif conf_type == "subtask_pick":
        # User replied with a number
        await messaging_client.send_message(
            chat_id,
            "Reply with the number of the task (e.g. `1`)"
        )
        # Re-store so next message picks up the number
        await context_service.set_pending_confirmation(user.id, "subtask_pick_number", data)

    elif conf_type == "project_confirm":
        await project_service.assign_project(data["message_id"], data["project"])
        await messaging_client.send_message(
            chat_id, f"📁 Added to project *{data['project']}*!"
        )


# ================== Helpers ==================

def _build_save_response(result: Dict) -> str:
    from zoneinfo import ZoneInfo

    category    = result.get("category", "Notes")
    all_buckets = result.get("all_buckets", [category])
    remind_at   = result.get("remind_at")
    due_date    = result.get("due_date")
    essence     = result.get("essence") or result.get("summary", "")
    split_count = result.get("split_count", 0)

    if remind_at:
        remind_dt = datetime.fromisoformat(remind_at).replace(
            tzinfo=ZoneInfo("UTC")
        ).astimezone(ZoneInfo("Asia/Kolkata"))
        time_str = remind_dt.strftime("%I:%M %p, %d %b")
        return f"✓ Saved & reminder set!\n\n⏰ {time_str} IST\n📝 {essence}"

    if split_count > 1:
        return f"✓ Saved {split_count} tasks!\n\n📝 {essence}"

    if "To-Do" in all_buckets and due_date:
        return f"✓ Added to To-Do!\n\n📝 {essence}\n📅 {due_date}"

    return f"✓ Saved to '{category}'!\n\n📝 {essence}"


def format_search_results(search_data) -> str:
    if isinstance(search_data, list):
        return "No results found." if not search_data else f"Found {len(search_data)} result(s)."
    natural = search_data.get("natural_response", "")
    results = search_data.get("results", [])
    if not results:
        return "I couldn't find anything related to that in your notes."
    if not natural or natural.strip() in ("{}", ""):
        return "\n".join(f"• {r['content']}" for r in results[:5])
    return natural


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


# ================== API Endpoints ==================

@app.post("/api/messages/capture")
async def capture_message(
    message: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await message_processor.process(
        user_phone=current_user.phone_number, content=message.content,
        message_type=message.message_type, media_url=message.media_url, db=db,
    )
    return {"success": True, "message": "Content captured successfully", "data": result}


@app.post("/api/search")
async def search_messages(
    search: SearchQuery,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    search_data = await search_service.search(
        user_phone=current_user.phone_number, query=search.query,
        limit=search.limit, category_filter=search.category_filter, db=db,
    )
    return {
        "success":          True,
        "query":            search.query,
        "natural_response": search_data.get("natural_response", ""),
        "results":          search_data.get("results", []),
        "total":            len(search_data.get("results", [])),
    }


@app.post("/api/categories/manage")
async def manage_categories(
    operation: CategoryOperation,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    phone = current_user.phone_number
    if operation.operation == "list":
        return {"success": True, "categories": await category_manager.list_categories(phone, db)}
    elif operation.operation == "create":
        return {"success": True, "data": await category_manager.create_category(phone, operation.category_name, operation.description, db)}
    elif operation.operation == "edit":
        return {"success": True, "data": await category_manager.edit_category(phone, operation.category_name, operation.new_name, operation.description, db)}
    elif operation.operation == "delete":
        return {"success": True, "data": await category_manager.delete_category(phone, operation.category_name, db)}
    raise HTTPException(status_code=400, detail="Invalid operation")


@app.get("/api/analytics")
async def get_user_analytics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select, func
    phone = current_user.phone_number
    total = await db.scalar(
        select(func.count(Message.id)).join(User).where(User.phone_number == phone)
    )
    cat_stats  = await db.execute(
        select(Category.name, func.count(Message.id))
        .join(Message.category).join(User).where(User.phone_number == phone)
        .group_by(Category.name)
    )
    type_stats = await db.execute(
        select(Message.message_type, func.count(Message.id))
        .join(User).where(User.phone_number == phone)
        .group_by(Message.message_type)
    )
    return {
        "total_messages": total,
        "by_category":    dict(cat_stats.all()),
        "by_type":        dict(type_stats.all()),
    }


@app.get("/api/webhook/info")
async def get_webhook_info():
    platform = Config.get_messaging_platform()
    if platform == MessagingPlatform.WHATSAPP:
        return {"platform": "whatsapp", "webhook_endpoint": "/webhook/whatsapp"}
    try:
        info = await messaging_client.get_webhook_info()
        return {
            "platform":        "telegram",
            "webhook_endpoint": "/webhook/telegram",
            "current_webhook": info.get("result", {}),
            "configured_url":  Config.TELEGRAM_WEBHOOK_URL,
        }
    except Exception:
        return {
            "platform":       "telegram",
            "configured_url": Config.TELEGRAM_WEBHOOK_URL,
            "status":         "not_configured",
        }
