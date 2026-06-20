"""
Extended Brain - Main FastAPI Application
Multi-platform messaging support (WhatsApp/Telegram)
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta, date
from enum import Enum
from contextlib import asynccontextmanager
from zoneinfo import ZoneInfo
import os
import re
import secrets
import asyncio
import httpx
from sqlalchemy import and_, or_, select, update, func, text, delete as sql_delete
from sqlalchemy.ext.asyncio import AsyncSession

from loguru import logger
from config import Config, MessagingPlatform
from messaging_factory import create_messaging_client, get_platform_name
from messaging_interface import MessagingClient

from database import get_db, init_db, engine, Base, async_session_maker, DeviceToken, StoredImage, GroupLastSeen, LabelAnnotation, IAPTransaction, PaymentOrder, AnalyticsEvent
from models import User, Message, Category, MessageType, ProAccount, ProAccountMember, Group, GroupMember, CouponCode, CouponRedemption
from services.group_service import group_service as grp_svc, total_unread_for_user
from services.coupon_service import coupon_service as cpn_svc
from services.payment_service import payment_service as pay_svc
from services.iap_service import iap_service
from cerebras_client import CerebrasClient
from services.message_processor import MessageProcessor
from services.search_service import SearchService
from services.category_manager import CategoryManager
from services.auth_service import AuthService, get_current_user, decode_user_id_safe
from services.reminder_service import ReminderService, send_apns_notification, Reminder
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

    # Load ONNX intent classifier in a thread so the event loop stays responsive
    # during the backbone download (can take 10–30s on first deploy).
    # Falls back to Gemini if the file is unavailable.
    from services.classifier_service import classifier_service
    loop = asyncio.get_event_loop()
    ok = await loop.run_in_executor(None, classifier_service.load)
    print(f"{'✓ Intent classifier loaded (ONNX)' if ok else '⚠ Intent classifier not loaded — using Gemini'}")

    from services.reminder_service import Reminder
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    scheduler_task = asyncio.create_task(_master_scheduler())
    print("✅ Master scheduler running")

    # Telegram bot is legacy/retired — we no longer register its webhook on startup.
    # (To fully stop an already-registered webhook, call deleteWebhook on the bot once.)

    print("✓ Extended Brain API started successfully")
    yield
    scheduler_task.cancel()
    print("✓ Extended Brain API shutdown")


async def _cleanup_stale_data():
    """Delete completed To-Do messages older than 7 days and fired one-time reminders older than 7 days."""
    cutoff = datetime.utcnow() - timedelta(days=7)
    async with async_session_maker() as db:
        # Find stale completed To-Do message IDs (done_at stored as ISO string in JSONB)
        stale_result = await db.execute(
            select(Message.id).where(
                Message.tags["primary_bucket"].astext == "To-Do",
                Message.tags["done"].astext == "true",
                Message.tags["done_at"].astext.isnot(None),
                text("(messages.tags->>'done_at')::timestamp < :cutoff").bindparams(cutoff=cutoff),
            )
        )
        stale_ids = [r[0] for r in stale_result.all()]

        if stale_ids:
            await db.execute(sql_delete(Reminder).where(Reminder.message_id.in_(stale_ids)))
            await db.execute(sql_delete(Recurrence).where(Recurrence.message_id.in_(stale_ids)))
            await db.execute(
                update(LabelAnnotation)
                .where(LabelAnnotation.message_id.in_(stale_ids))
                .values(message_id=None)
            )
            await db.execute(sql_delete(Message).where(Message.id.in_(stale_ids)))
            await db.commit()
            print(f"[cleanup] Deleted {len(stale_ids)} completed To-Do messages older than 7 days")

        # Delete fired one-time reminders (recurrence IS NULL = not recurring) older than 7 days
        fired_result = await db.execute(
            sql_delete(Reminder).where(
                Reminder.is_sent == True,
                Reminder.recurrence.is_(None),
                Reminder.sent_at < cutoff,
            )
        )
        await db.commit()
        fired_count = fired_result.rowcount
        if fired_count:
            print(f"[cleanup] Deleted {fired_count} fired one-time reminders older than 7 days")


async def _master_scheduler():
    """Single scheduler loop — runs every 60 seconds."""
    print("[scheduler] Master scheduler started")
    tick = 0
    while True:
        try:
            # Do work first so reminders fire immediately on startup, then sleep
            tick += 1
            await reminder_service.run_scheduler_tick()
            await recurrence_service.run()
            await briefing_service.run()

            if tick % 30 == 0:
                await nudge_service.run_idle_nudges()
                await nudge_service.run_followup_checks()

            # Run stale data cleanup once at startup and then every 24 hours
            if tick == 1 or tick % 1440 == 0:
                await _cleanup_stale_data()

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[scheduler] Error: {e}")
        finally:
            await asyncio.sleep(60)


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
        "https://extendedmindsai.com",
        "https://www.extendedmindsai.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Services — ORDER MATTERS
cerebras_client    = CerebrasClient(provider="gemini")
messaging_client: MessagingClient = create_messaging_client()
search_service     = SearchService(cerebras_client)
category_manager   = CategoryManager(cerebras_client)
auth_service       = AuthService(messaging_client)
reminder_service   = ReminderService(cerebras_client)
message_processor  = MessageProcessor(cerebras_client, reminder_service=reminder_service)
project_service    = ProjectService(cerebras_client)
subtask_service    = SubtaskService(cerebras_client)
recurrence_service = RecurrenceService(cerebras_client)


# ================== WebSocket Connection Manager ==================

class WSConnectionManager:
    """Tracks live WebSocket connections per group, broadcasts new messages instantly."""

    def __init__(self):
        # group_id → { user_id → set[WebSocket] }
        self._conns: dict[int, dict[int, set]] = {}

    async def connect(self, ws: WebSocket, group_id: int, user_id: int) -> None:
        await ws.accept()
        self._conns.setdefault(group_id, {}).setdefault(user_id, set()).add(ws)

    def disconnect(self, ws: WebSocket, group_id: int, user_id: int) -> None:
        group = self._conns.get(group_id, {})
        sockets = group.get(user_id, set())
        sockets.discard(ws)
        if not sockets:
            group.pop(user_id, None)

    async def broadcast(self, group_id: int, data: dict, exclude_user_id: Optional[int] = None) -> None:
        for uid, sockets in list(self._conns.get(group_id, {}).items()):
            if uid == exclude_user_id:
                continue
            for ws in list(sockets):
                try:
                    await ws.send_json(data)
                except Exception:
                    sockets.discard(ws)


ws_manager = WSConnectionManager()


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
    phone_number: str      = Field(..., min_length=10, max_length=20)
    password:     str      = Field(..., min_length=6)
    timezone:     Optional[str] = "Asia/Kolkata"


class OTPSendRequest(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=20)


class OTPVerifyRequest(BaseModel):
    phone_number: str = Field(..., min_length=10, max_length=20)
    otp:          str = Field(..., min_length=6, max_length=6)


class LoginRequest(BaseModel):
    phone_number: Optional[str] = None
    email:        Optional[str] = None
    password:     str


class ForgotPasswordRequest(BaseModel):
    phone_number: str
    new_password: str = Field(..., min_length=6)
    otp: Optional[str] = None              # required when ENABLE_OTP (MSG91 path)


class FirebaseVerifyPhoneRequest(BaseModel):
    id_token: str = Field(..., min_length=1)


class FirebaseResetPasswordRequest(BaseModel):
    id_token:     str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=6)


class AppleSignInRequest(BaseModel):
    id_token:  str = Field(..., min_length=1)
    full_name: Optional[str] = None


class OAuthVerifyPhoneRequest(BaseModel):
    session_token: str
    # Phone verification — two supported paths during the Firebase→MSG91 transition:
    #   MSG91/OTP: phone_number + otp   ·   Legacy Firebase: phone_id_token
    phone_id_token: Optional[str] = None
    phone_number:   Optional[str] = None
    otp:            Optional[str] = None

class GoogleSignInRequest(BaseModel):
    id_token:  str = Field(..., min_length=1)
    full_name: Optional[str] = None


class TelegramLinkRequest(BaseModel):
    phone_number:     str
    telegram_chat_id: str


class MessageCreate(BaseModel):
    content:          str
    message_type:     MessageTypeEnum = MessageTypeEnum.TEXT
    media_url:        Optional[str]   = None
    metadata:         Optional[Dict[str, Any]] = None
    group_id:         Optional[int]   = None
    expense_amount:   Optional[float] = None
    expense_category: Optional[str]   = None
    expense_payer_id:   Optional[int] = None
    expense_payer_name: Optional[str] = None
    expense_context:  Optional[str]   = None
    force_bucket:     Optional[str]   = None


class SearchQuery(BaseModel):
    query:           str
    limit:           int = 10
    category_filter: Optional[List[str]] = None
    group_id:        Optional[int] = None
    fast:            bool = False  # skip LLM; return embedding+keyword results instantly


class DoneRequest(BaseModel):
    done: bool = True


class RemindAtRequest(BaseModel):
    remind_at: str  # ISO 8601 datetime with timezone, e.g. "2026-06-02T18:00:00+05:30"


class CategoryOperation(BaseModel):
    operation:     str
    category_name: Optional[str] = None
    new_name:      Optional[str] = None
    description:   Optional[str] = None


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
    from services.classifier_service import classifier_service
    return {
        "status":      "healthy",
        "timestamp":   datetime.now().isoformat(),
        "platform":    get_platform_name(),
        "classifier":  {
            "ready":   classifier_service.is_ready,
            "classes": list(classifier_service._classes) if classifier_service.is_ready else [],
        },
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


@app.post("/api/auth/firebase-verify-phone")
async def firebase_verify_phone(request: FirebaseVerifyPhoneRequest, db: AsyncSession = Depends(get_db)):
    """
    Verify a Firebase phone auth ID token (issued by the iOS app after SMS verification).
    Stores a verified OTPVerification record so the subsequent /api/users/register call
    sees the phone as confirmed (when ENABLE_OTP=true).
    """
    from services.firebase_service import verify_phone_token
    from database import OTPVerification
    logger.info("[auth/firebase-verify-phone] request received")
    try:
        phone_number = verify_phone_token(request.id_token)
    except ValueError as e:
        logger.warning("[auth/firebase-verify-phone] token verification FAILED — {}", e)
        raise HTTPException(status_code=400, detail=str(e))

    # Remove any existing OTP records for this phone and insert a pre-verified one.
    existing = await db.execute(select(OTPVerification).where(OTPVerification.phone_number == phone_number))
    for row in existing.scalars():
        await db.delete(row)

    otp_record = OTPVerification(
        phone_number=phone_number,
        otp_code="firebase",
        is_verified=True,
        expires_at=datetime.utcnow() + timedelta(minutes=10),
    )
    db.add(otp_record)
    await db.commit()

    logger.info("[auth/firebase-verify-phone] OK — phone=***{}", phone_number[-4:])
    return {"success": True, "verified": True, "phone_number": phone_number}


@app.post("/api/auth/apple")
async def apple_sign_in(request: AppleSignInRequest, db: AsyncSession = Depends(get_db)):
    """Sign in or register via Apple ID (Firebase-verified)."""
    from services.firebase_service import verify_apple_token
    try:
        uid, email, firebase_name = verify_apple_token(request.id_token)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Apple sign-in Firebase error: {e}")
        raise HTTPException(status_code=500, detail="Apple sign-in verification failed")

    display_name = request.full_name or firebase_name or (email.split("@")[0] if email else "Apple User")

    # 1. Returning Apple user (already linked)
    user = await db.scalar(select(User).where(User.apple_uid == uid))

    # 2. Existing account with same email — link the Apple UID
    if not user and email:
        user = await db.scalar(select(User).where(User.email == email))
        if user:
            user.apple_uid = uid

    # 3. No account found — need phone to link or create
    if not user:
        logger.info("[auth/apple] uid={}... no account found — returning needs_phone", uid[:8])
        session_token = auth_service.create_oauth_session_token(
            provider="apple", uid=uid, email=email or "", name=display_name
        )
        return {"success": True, "needs_phone": True, "session_token": session_token}

    logger.info("[auth/apple] uid={}... matched user_id={}", uid[:8], user.id)
    user.last_login = datetime.utcnow()
    await db.commit()

    token = auth_service.create_access_token(user.id)
    return {
        "success": True,
        "data": {
            "access_token": token,
            "user": {
                "id": user.id,
                "phone_number": user.phone_number,
                "name": user.name,
                "email": user.email,
                "timezone": user.timezone,
            },
        },
    }


async def _resolve_verified_phone(request: OAuthVerifyPhoneRequest, db: AsyncSession) -> str:
    """
    Resolve and verify the phone number for an OAuth link-phone request.
    Supports both paths during the Firebase→MSG91 transition:
      • MSG91/OTP:  {phone_number, otp}  — verified against otp_verifications
      • Legacy:     {phone_id_token}     — verified Firebase phone token
    Returns the verified phone number, or raises HTTPException(400).
    """
    if request.phone_number and request.otp:
        result = await auth_service.verify_otp(request.phone_number, request.otp, db)
        if not result.get("verified"):
            raise HTTPException(status_code=400, detail=result.get("message", "Invalid OTP"))
        return request.phone_number
    if request.phone_id_token:
        from services.firebase_service import verify_phone_token
        try:
            return verify_phone_token(request.phone_id_token)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    raise HTTPException(status_code=400, detail="Phone verification required (otp or phone_id_token)")


@app.post("/api/auth/apple/verify-phone")
async def apple_verify_phone(request: OAuthVerifyPhoneRequest, db: AsyncSession = Depends(get_db)):
    """Complete Apple Sign-In by verifying phone — links to existing account or creates new one."""
    logger.info("[auth/apple/verify-phone] request received")
    try:
        session = auth_service.decode_oauth_session_token(request.session_token)
    except ValueError as e:
        logger.warning("[auth/apple/verify-phone] session token INVALID — {}", e)
        raise HTTPException(status_code=400, detail=str(e))
    phone_number = await _resolve_verified_phone(request, db)

    uid   = session["uid"]
    email = session["email"]
    name  = session["name"]
    logger.info("[auth/apple/verify-phone] uid={}... email={} phone=***{}", uid[:8], email, phone_number[-4:])

    user = await db.scalar(select(User).where(User.phone_number == phone_number))
    if user:
        logger.info("[auth/apple/verify-phone] found existing user by phone — user_id={}, linking apple_uid", user.id)
        user.apple_uid = uid
    else:
        if email:
            user = await db.scalar(select(User).where(User.email == email))
        if user:
            logger.info("[auth/apple/verify-phone] found existing user by email — user_id={}, linking apple_uid", user.id)
            user.apple_uid = uid
        else:
            logger.info("[auth/apple/verify-phone] no existing user found — creating new account")
            user = User(
                phone_number=phone_number,
                email=email or f"{uid}@apple.extendedminds",
                name=name,
                age=0,
                password_hash="apple_oauth_no_password",
                apple_uid=uid,
            )
            db.add(user)
            await db.flush()

    user.last_login = datetime.utcnow()
    await db.commit()

    logger.info("[auth/apple/verify-phone] OK — user_id={}", user.id)
    token = auth_service.create_access_token(user.id)
    return {
        "success": True,
        "data": {
            "access_token": token,
            "user": {
                "id": user.id,
                "phone_number": user.phone_number,
                "name": user.name,
                "email": user.email,
                "timezone": user.timezone,
            },
        },
    }


@app.post("/api/auth/google")
async def google_sign_in(request: GoogleSignInRequest, db: AsyncSession = Depends(get_db)):
    """Sign in or register via Google Account (Firebase-verified)."""
    from services.firebase_service import verify_google_token
    try:
        uid, email, firebase_name = verify_google_token(request.id_token)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Google sign-in Firebase error: {e}")
        raise HTTPException(status_code=500, detail="Google sign-in verification failed")

    display_name = request.full_name or firebase_name or (email.split("@")[0] if email else "Google User")

    # 1. Returning Google user (already linked)
    user = await db.scalar(select(User).where(User.google_uid == uid))

    # 2. Existing account with same email — link the Google UID
    if not user and email:
        user = await db.scalar(select(User).where(User.email == email))
        if user:
            user.google_uid = uid

    # 3. No account found — need phone to link or create
    if not user:
        logger.info("[auth/google] uid={}... no account found — returning needs_phone", uid[:8])
        session_token = auth_service.create_oauth_session_token(
            provider="google", uid=uid, email=email or "", name=display_name
        )
        return {"success": True, "needs_phone": True, "session_token": session_token}

    logger.info("[auth/google] uid={}... matched user_id={}", uid[:8], user.id)
    user.last_login = datetime.utcnow()
    await db.commit()

    token = auth_service.create_access_token(user.id)
    return {
        "success": True,
        "data": {
            "access_token": token,
            "user": {
                "id":           user.id,
                "phone_number": user.phone_number,
                "name":         user.name,
                "email":        user.email,
                "timezone":     user.timezone,
            },
        },
    }


@app.post("/api/auth/google/verify-phone")
async def google_verify_phone(request: OAuthVerifyPhoneRequest, db: AsyncSession = Depends(get_db)):
    """Complete Google Sign-In by verifying phone — links to existing account or creates new one."""
    logger.info("[auth/google/verify-phone] request received")
    try:
        session = auth_service.decode_oauth_session_token(request.session_token)
    except ValueError as e:
        logger.warning("[auth/google/verify-phone] session token INVALID — {}", e)
        raise HTTPException(status_code=400, detail=str(e))
    phone_number = await _resolve_verified_phone(request, db)

    uid   = session["uid"]
    email = session["email"]
    name  = session["name"]
    logger.info("[auth/google/verify-phone] uid={}... email={} phone=***{}", uid[:8], email, phone_number[-4:])

    user = await db.scalar(select(User).where(User.phone_number == phone_number))
    if user:
        logger.info("[auth/google/verify-phone] found existing user by phone — user_id={}, linking google_uid", user.id)
        user.google_uid = uid
    else:
        if email:
            user = await db.scalar(select(User).where(User.email == email))
        if user:
            logger.info("[auth/google/verify-phone] found existing user by email — user_id={}, linking google_uid", user.id)
            user.google_uid = uid
        else:
            logger.info("[auth/google/verify-phone] no existing user found — creating new account")
            user = User(
                phone_number=phone_number,
                email=email or f"{uid}@google.extendedminds",
                name=name,
                age=0,
                password_hash="google_oauth_no_password",
                google_uid=uid,
            )
            db.add(user)
            await db.flush()

    user.last_login = datetime.utcnow()
    await db.commit()
    logger.info("[auth/google/verify-phone] OK — user_id={}", user.id)

    token = auth_service.create_access_token(user.id)
    return {
        "success": True,
        "data": {
            "access_token": token,
            "user": {
                "id":           user.id,
                "phone_number": user.phone_number,
                "name":         user.name,
                "email":        user.email,
                "timezone":     user.timezone,
            },
        },
    }


@app.post("/api/auth/firebase-reset-password")
async def firebase_reset_password(request: FirebaseResetPasswordRequest, db: AsyncSession = Depends(get_db)):
    """
    Verify Firebase phone token then reset the password for that phone number atomically.
    Replaces the unauthenticated /api/auth/forgot-password flow with a Firebase-gated one.
    """
    from services.firebase_service import verify_phone_token
    try:
        phone_number = verify_phone_token(request.id_token)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = await auth_service.reset_password(
        phone_number=phone_number, new_password=request.new_password, db=db
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/auth/login")
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    if not request.email and not request.phone_number:
        raise HTTPException(status_code=422, detail="Provide email or phone_number")
    result = await auth_service.login_user(
        password=request.password,
        db=db,
        email=request.email,
        phone_number=request.phone_number,
    )
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["message"])
    return result


@app.post("/api/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)):
    # When OTP is enabled, require a verified OTP for this phone before resetting.
    if Config.ENABLE_OTP:
        if not request.otp:
            raise HTTPException(status_code=400, detail="OTP required")
        verify = await auth_service.verify_otp(request.phone_number, request.otp, db)
        if not verify.get("verified"):
            raise HTTPException(status_code=400, detail=verify.get("message", "Invalid OTP"))
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
    """
    Permanently delete the account and all data the user owns.

    Deletes in FK-dependency order (most rows have no ON DELETE CASCADE at the
    DB level, so a bare `db.delete(user)` would fail for any non-trivial user):
      1. Reassign cross-account ownership refs (groups/invites this user created
         inside SOMEONE ELSE'S Pro account) to that account's owner.
      2. Purge single-user-owned rows that FK to users.id / messages.id.
      3. Tear down the user's OWN Pro account (cascades members + groups).
      4. Delete the user's own messages everywhere (personal + group-authored).
      5. Delete the user (cascades categories + group memberships).
    """
    uid = current_user.id

    owned_account = await db.scalar(select(ProAccount).where(ProAccount.owner_id == uid))
    owned_account_id = owned_account.id if owned_account else None

    # 1. Reassign cross-account references via raw UPDATE so the ORM session
    #    doesn't get confused trying to flush related ProAccount objects.
    #
    #    Groups this user created inside SOMEONE ELSE'S account: hand to that owner.
    other_account_ids_groups = (await db.execute(
        select(ProAccount.id, ProAccount.owner_id)
        .join(Group, Group.account_id == ProAccount.id)
        .where(Group.created_by == uid)
        .where(ProAccount.owner_id != uid)
    )).all()
    for acct_id, owner_id in other_account_ids_groups:
        await db.execute(
            update(Group)
            .where(Group.account_id == acct_id, Group.created_by == uid)
            .values(created_by=owner_id)
        )

    #    Invites this user sent inside SOMEONE ELSE'S account: hand to that owner.
    other_account_ids_invites = (await db.execute(
        select(ProAccount.id, ProAccount.owner_id)
        .join(ProAccountMember, ProAccountMember.account_id == ProAccount.id)
        .where(ProAccountMember.invited_by == uid)
        .where(ProAccount.owner_id != uid)
    )).all()
    for acct_id, owner_id in other_account_ids_invites:
        await db.execute(
            update(ProAccountMember)
            .where(ProAccountMember.account_id == acct_id, ProAccountMember.invited_by == uid)
            .values(invited_by=owner_id)
        )

    # 2. Purge dependent rows that FK → messages.id or → users.id.
    #
    #    Reminders and Recurrences are tricky: another user can hold a reminder
    #    that references a message THIS user authored (e.g. a group member set a
    #    reminder on this user's group capture). We must delete BOTH:
    #      a) rows owned by this user (user_id == uid)
    #      b) rows pointing at any message authored by this user (message_id IN …)
    #    Do (b) first so (a) is a no-op subset rather than a duplicate condition.
    user_msg_ids_subq = select(Message.id).where(Message.user_id == uid).scalar_subquery()
    for stmt in (
        sql_delete(Recurrence).where(Recurrence.message_id.in_(user_msg_ids_subq)),
        sql_delete(Recurrence).where(Recurrence.user_id == uid),
        sql_delete(Reminder).where(Reminder.message_id.in_(user_msg_ids_subq)),
        sql_delete(Reminder).where(Reminder.user_id == uid),
        sql_delete(LabelAnnotation).where(LabelAnnotation.user_id == uid),
        sql_delete(DeviceToken).where(DeviceToken.user_id == uid),
        sql_delete(StoredImage).where(StoredImage.user_id == uid),
        sql_delete(IAPTransaction).where(IAPTransaction.user_id == uid),
        sql_delete(PaymentOrder).where(PaymentOrder.user_id == uid),
        sql_delete(CouponRedemption).where(CouponRedemption.user_id == uid),
        sql_delete(GroupLastSeen).where(GroupLastSeen.user_id == uid),
    ):
        await db.execute(stmt)

    # 3. Tear down the user's own Pro account: wipe its groups' shared content, then
    #    the groups, members, and the account itself. Explicit bulk deletes (not
    #    `db.delete(instance)`) avoid async lazy-loading of cascade relationships.
    if owned_account_id is not None:
        group_ids = (await db.execute(
            select(Group.id).where(Group.account_id == owned_account_id)
        )).scalars().all()
        if group_ids:
            group_msg_ids_subq = select(Message.id).where(Message.group_id.in_(group_ids)).scalar_subquery()
            await db.execute(sql_delete(Recurrence).where(Recurrence.message_id.in_(group_msg_ids_subq)))
            await db.execute(sql_delete(Reminder).where(Reminder.message_id.in_(group_msg_ids_subq)))
            await db.execute(sql_delete(Message).where(Message.group_id.in_(group_ids)))
            await db.execute(sql_delete(GroupLastSeen).where(GroupLastSeen.group_id.in_(group_ids)))
            await db.execute(sql_delete(GroupMember).where(GroupMember.group_id.in_(group_ids)))
        await db.execute(sql_delete(Group).where(Group.account_id == owned_account_id))
        await db.execute(sql_delete(ProAccountMember).where(ProAccountMember.account_id == owned_account_id))
        await db.execute(sql_delete(ProAccount).where(ProAccount.id == owned_account_id))

    # 4. Delete the user's own messages everywhere (personal + authored in groups
    #    owned by others). Reminders/annotations referencing them are already gone.
    await db.execute(sql_delete(Message).where(Message.user_id == uid))

    # 5. Delete the user's remaining owned rows, then the user itself.
    await db.execute(sql_delete(Category).where(Category.user_id == uid))
    await db.execute(sql_delete(GroupMember).where(GroupMember.user_id == uid))
    await db.execute(sql_delete(User).where(User.id == uid))
    await db.commit()
    return {"success": True, "message": "Account deleted."}


@app.post("/api/auth/link-telegram")
async def link_telegram(request: TelegramLinkRequest, db: AsyncSession = Depends(get_db)):
    user = await db.scalar(select(User).where(User.phone_number == request.phone_number))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.telegram_chat_id = request.telegram_chat_id
    await db.commit()
    return {"success": True, "message": "Telegram linked successfully"}


@app.get("/api/messages/detail/{message_id}")
async def get_message(
    message_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = await db.execute(
        select(Message, Category)
        .outerjoin(Category, Message.category_id == Category.id)
        .where(and_(Message.id == message_id, Message.user_id == current_user.id))
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Message not found")
    msg, cat = row
    tags   = msg.tags if isinstance(msg.tags, dict) else {}
    bucket = tags.get("primary_bucket") or tags.get("intent_bucket") or (cat.name if cat else "Random")
    raw_items = tags.get("subtasks", [])
    items = [{"task": s["task"], "done": s.get("done", False)}
             for s in raw_items if isinstance(s, dict) and "task" in s]
    return {
        "success": True,
        "data": {
            "id":                  msg.id,
            "content":             msg.content,
            "essence":             msg.summary,
            "category":            bucket,
            "tags":                tags,
            "items":               items,
            "message_type":        msg.message_type.value if msg.message_type else "text",
            "created_at":          msg.created_at.isoformat(),
            "group_id":            msg.group_id,
            "sender_id":           msg.user_id,
            "assigned_to_user_id": msg.assigned_to_user_id,
        }
    }


@app.get("/api/users/me")
async def get_me(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    last_capture = await db.scalar(
        select(func.max(Message.created_at)).where(
            Message.user_id == current_user.id,
            Message.group_id.is_(None),
            Message.assigned_to_user_id.is_(None),
        )
    )
    return {
        "success": True,
        "data": {
            "id": current_user.id,
            "name": current_user.name,
            "email": current_user.email,
            "phone_number": current_user.phone_number,
            "timezone": current_user.timezone,
            "is_pro": current_user.is_pro,
            "briefing_time": current_user.briefing_time,
            "last_capture_at": last_capture.isoformat() if last_capture else None,
        }
    }


@app.patch("/api/users/me")
async def update_profile(
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="name cannot be empty")
    current_user.name = name
    await db.commit()
    return {
        "success": True,
        "data": {
            "id":           current_user.id,
            "name":         current_user.name,
            "email":        current_user.email,
            "phone_number": current_user.phone_number,
            "timezone":     current_user.timezone,
            "is_pro":       current_user.is_pro,
            "briefing_time": current_user.briefing_time,
        }
    }


@app.post("/api/users/device-token")
async def register_device_token(
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    token = (body.get("device_token") or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="device_token required")

    existing = await db.scalar(select(DeviceToken).where(DeviceToken.token == token))
    if existing:
        existing.user_id    = current_user.id
        existing.updated_at = datetime.utcnow()
    else:
        db.add(DeviceToken(user_id=current_user.id, token=token, platform="ios"))
    await db.commit()
    return {"success": True}


@app.patch("/api/users/briefing")
async def update_briefing_settings(
    body: dict,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update morning briefing preferences from iOS.
    Body: {"enabled": bool, "time": "HH:MM"}
    Setting enabled=false stores null briefing_time (scheduler skips nulls).
    """
    enabled = body.get("enabled", True)
    time_str = body.get("time", "08:00")

    if not enabled:
        new_time = None
    else:
        # Validate HH:MM format
        import re
        if not re.match(r"^\d{2}:\d{2}$", str(time_str)):
            raise HTTPException(status_code=400, detail="time must be HH:MM format")
        new_time = time_str

    await db.execute(
        update(User).where(User.id == current_user.id).values(briefing_time=new_time)
    )
    await db.commit()
    return {"success": True, "briefing_time": new_time}


@app.delete("/api/users/me/data")
async def reset_personal_data(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete all of the user's captured content — personal AND the messages they
    authored inside groups. Preserves the account, Pro status, and group
    memberships themselves (other members' messages are untouched).
    """
    # Recurrences and Reminders both FK to messages.id — delete before messages.
    # Unlink annotations (preserve training data) rather than deleting them.
    await db.execute(
        sql_delete(Recurrence).where(Recurrence.user_id == current_user.id)
    )
    await db.execute(
        sql_delete(Reminder).where(Reminder.user_id == current_user.id)
    )
    await db.execute(
        update(LabelAnnotation).where(LabelAnnotation.user_id == current_user.id).values(message_id=None)
    )
    # All messages this user authored, in personal feed and in any group.
    await db.execute(
        sql_delete(Message).where(Message.user_id == current_user.id)
    )
    await db.commit()
    return {"success": True, "message": "Brain data cleared."}


@app.post("/api/users/register")
async def register_user(user_data: UserRegistrationRequest, db: AsyncSession = Depends(get_db)):
    result = await auth_service.register_user(
        phone_number=user_data.phone_number, name=user_data.name,
        email=user_data.email, age=user_data.age,
        password=user_data.password, timezone=user_data.timezone, db=db,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


# ================== Telegram Webhook ==================

@app.post("/webhook/telegram")
async def handle_telegram_webhook(
    request: Request, background_tasks: BackgroundTasks
):
    if Config.get_messaging_platform() != MessagingPlatform.TELEGRAM:
        raise HTTPException(status_code=400, detail="Telegram not configured")

    webhook_data = await request.json()

    if "callback_query" in webhook_data:
        # Callbacks are handled inline with their own session — not in a background task
        # because Telegram expects answerCallbackQuery within a few seconds
        asyncio.create_task(_handle_callback_query(webhook_data["callback_query"]))
        return {"ok": True}

    # FIX: do NOT pass db — background task opens its own session
    background_tasks.add_task(process_webhook_message, webhook_data)
    return {"ok": True}


# ================== Callback Handler ==================

async def _handle_callback_query(callback: Dict):
    """Handles inline keyboard callbacks. Opens its own DB session."""
    callback_id   = callback["id"]
    chat_id       = str(callback["from"]["id"])
    callback_data = callback.get("data", "")
    tg_message_id = callback["message"]["message_id"]

    # ── done:<id> ─────────────────────────────────────────────────
    if callback_data.startswith("done:"):
        msg_db_id = int(callback_data.split(":")[1])
        async with async_session_maker() as session:
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
        parts   = callback_data.split(":")
        msg_id  = int(parts[1])
        minutes = int(parts[2]) if len(parts) > 2 else 1440
        await nudge_service.snooze_message(msg_id, minutes)

        if minutes < 60:
            snooze_label = f"{minutes} min"
        elif minutes == 60:
            snooze_label = "1 hr"
        else:
            snooze_label = f"{minutes // 60} hrs"

        await _answer_callback(callback_id, f"⏰ Snoozed {snooze_label}")

        # Remove buttons after snooze so message stays clean
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/editMessageReplyMarkup",
                json={
                    "chat_id":      chat_id,
                    "message_id":   tg_message_id,
                    "reply_markup": {"inline_keyboard": []},
                },
            )

    # ── subtask:<message_id>:<index> ──────────────────────────────
    elif callback_data.startswith("subtask:"):
        parts  = callback_data.split(":")
        msg_id = int(parts[1])
        idx    = int(parts[2])
        await subtask_service.complete_subtask(msg_id, idx)
        await _answer_callback(callback_id, "✓ Subtask done!")
        await _refresh_subtask_message(chat_id, tg_message_id, msg_id)

    # ── list_done:<message_id>:<index> ────────────────────────────
    elif callback_data.startswith("list_done:"):
        parts  = callback_data.split(":")
        msg_id = int(parts[1])
        idx    = int(parts[2])
        from services.list_service import ListService
        ls = ListService(cerebras_client)
        await ls.complete_item(msg_id, idx)
        await _answer_callback(callback_id, "✓ Done!")
        await _refresh_list_message(chat_id, tg_message_id, msg_id)

    # ── list_clear:<message_id> ───────────────────────────────────
    elif callback_data.startswith("list_clear:"):
        msg_id = int(callback_data.split(":")[1])
        from services.list_service import ListService
        ls = ListService(cerebras_client)
        removed = await ls.clear_done_items(msg_id)
        await _answer_callback(callback_id, f"🗑 Cleared {removed} done items")
        await _refresh_list_message(chat_id, tg_message_id, msg_id)

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

    # ── proj_yes:<msg_id>:<project> ───────────────────────────────
    elif callback_data.startswith("proj_yes:"):
        parts     = callback_data.split(":", 2)
        msg_id    = int(parts[1])
        proj_name = parts[2]
        await project_service.assign_project(msg_id, proj_name)
        await _answer_callback(callback_id, f"📁 Added to {proj_name}")

        # FIX: remove the Yes/No buttons and update message text
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/editMessageText",
                json={
                    "chat_id":      chat_id,
                    "message_id":   tg_message_id,
                    "text":         f"📁 Added to project *{proj_name}*",
                    "parse_mode":   "Markdown",
                    "reply_markup": {"inline_keyboard": []},  # clears buttons
                },
            )

        async with async_session_maker() as session:
            user = await session.scalar(select(User).where(User.telegram_chat_id == chat_id))
            if user:
                await context_service.clear_pending_confirmation(user.id)

    elif callback_data.startswith("proj_no:"):
        await _answer_callback(callback_id, "OK, skipped")

        # FIX: remove the Yes/No buttons
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/editMessageText",
                json={
                    "chat_id":      chat_id,
                    "message_id":   tg_message_id,
                    "text":         "📁 _Not added to any project_",
                    "parse_mode":   "Markdown",
                    "reply_markup": {"inline_keyboard": []},  # clears buttons
                },
            )

        async with async_session_maker() as session:
            user = await session.scalar(select(User).where(User.telegram_chat_id == chat_id))
            if user:
                await context_service.clear_pending_confirmation(user.id)


    elif callback_data == "noop":
        # Section divider button — acknowledge silently
        await _answer_callback(callback_id, "")

    else:
        await _answer_callback(callback_id, "Unknown action")


async def _answer_callback(callback_id: str, text: str = ""):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/answerCallbackQuery",
            json={"callback_query_id": callback_id, "text": text},
        )


# ================== Message refresh helpers ==================

async def _refresh_checklist_message(chat_id: str, tg_message_id: int):
    async with async_session_maker() as session:
        user = await session.scalar(
            select(User).where(User.telegram_chat_id == chat_id)
        )
        if not user:
            return

        # Retrieve the date this checklist was originally showing
        ctx       = await context_service.get_checklist_context(user.id, tg_message_id)
        date_from = ctx.get("date_from") if ctx else None
        date_to   = ctx.get("date_to")   if ctx else None

        # If no context (old message), default to today
        from datetime import date as date_type
        today_str = str(date_type.today())
        date_from = date_from or today_str
        date_to   = date_to   or today_str

        # Fetch directly from DB — don't go through LLM search
        from services.search_service import SearchService
        todo_result = await search_service._fetch_todos_direct(
            user_id=user.id,
            date_from=date_from,
            date_to=date_to,
            db=session,
        )

    results = [
        r for r in todo_result.get("results", [])
        if not r.get("tags", {}).get("done", False)
    ]

    text, reply_markup = format_todo_checklist(results, date_from, date_to)
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


async def _refresh_list_message(chat_id: str, tg_message_id: int, message_id: int):
    async with async_session_maker() as session:
        msg = await session.scalar(select(Message).where(Message.id == message_id))
        if not msg:
            return
    from services.list_service import ListService
    ls = ListService(cerebras_client)
    text, reply_markup = ls.format_for_telegram(msg)
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

def format_todo_checklist(
    results: List[Dict],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> tuple[str, dict]:
    today     = date.today()
    yesterday = today - timedelta(days=1)

    if date_from and date_to:
        if date_from == date_to:
            d = datetime.fromisoformat(date_from).date()
            if d == today:        label = "today"
            elif d == yesterday:  label = "yesterday"
            else:                 label = d.strftime("%a, %d %b")
        else:
            label = f"{date_from} → {date_to}"
    else:
        label = "today"

    text    = f"📋 *To-Do — {label}*\n\n"
    buttons = []

    if not results:
        text += "_Nothing here yet!_"
        return text, {"inline_keyboard": buttons}

    # Separate overdue from current
    current = [r for r in results if not r.get("is_overdue")]
    overdue = [r for r in results if r.get("is_overdue")]

    # ── Group current items by split_from ────────────────────────
    # Items with same split_from = came from same batch message
    # Items with no split_from = standalone tasks
    groups: Dict[str, List[Dict]] = {}   # split_from → [items]
    standalone: List[Dict] = []

    for item in current:
        split_from = item.get("split_from", "").strip()
        if split_from:
            if split_from not in groups:
                groups[split_from] = []
            groups[split_from].append(item)
        else:
            standalone.append(item)

    # Groups with only 1 item — treat as standalone (no point grouping)
    for split_from, items in list(groups.items()):
        if len(items) == 1:
            standalone.append(items[0])
            del groups[split_from]

    # ── Render grouped items ──────────────────────────────────────
    for split_from, items in groups.items():
        # Derive group title from split_from — first line, cleaned
        raw_title  = split_from.split("\n")[0].strip().rstrip(":")
        # Strip bullet markers if present
        raw_title  = re.sub(r"^[-*•]\s*", "", raw_title)
        group_title = raw_title[:40] if raw_title else "Tasks"
        done_count  = sum(1 for i in items if i.get("tags", {}).get("done"))
        total_count = len(items)
        all_done    = done_count == total_count

        if all_done:
            # Strike through the whole group header
            text += f"~✓ {group_title} ({total_count})~\n"
        else:
            # Group header button — shows title + count, marks ALL done on tap
            # Use first item's id as representative (or handle noop)
            buttons.append([{
                "text": f"📌 {group_title}  {done_count}/{total_count}",
                "callback_data": f"noop",
            }])
            # Sub-items — indented visually with a leader character
            for item in items:
                tags     = item.get("tags", {})
                is_done  = tags.get("done", False)
                due_time = item.get("event_time") or tags.get("event_time", "")
                time_str = f" _{due_time}_" if due_time else ""
                label    = (
                    item.get("essence")
                    or item.get("content", "").split("\n")[0]
                ).strip()[:38]

                if not is_done:
                    buttons.append([{
                        "text": f"-- {label}{time_str}",
                        "callback_data": f"done:{item['id']}",
                    }])
                else:
                    # Done sub-items shown in text, not as buttons
                    text += f"  ~✓ {label}~\n"

    # ── Render standalone items ───────────────────────────────────
    for item in standalone:
        tags     = item.get("tags", {})
        is_done  = tags.get("done", False)
        due_time = item.get("event_time") or tags.get("event_time", "")
        time_str = f" _{due_time}_" if due_time else ""
        has_subs = bool(tags.get("subtasks"))
        sub_icon = " 📎" if has_subs else ""
        label    = (
            item.get("essence")
            or item.get("content", "").split("\n")[0]
        ).strip()[:42]

        if is_done:
            text += f"~✓ {label}{time_str}~\n"
        else:
            buttons.append([{
                "text":          f"📌 {label}{time_str}{sub_icon}",
                "callback_data": f"done:{item['id']}",
            }])

    # ── Render overdue section ────────────────────────────────────
    if overdue:
        buttons.append([{
            "text":          "⚠ Overdue",
            "callback_data": "noop",
        }])
        for item in overdue:
            tags  = item.get("tags", {})
            due   = item.get("due_date", "")
            try:
                due_label = datetime.fromisoformat(due).strftime("%d %b")
            except Exception:
                due_label = due
            label = (
                item.get("essence")
                or item.get("content", "").split("\n")[0]
            ).strip()[:36]

            if not item.get("tags", {}).get("done"):
                buttons.append([{
                    "text":          f"☐ {label} _{due_label}_",
                    "callback_data": f"done:{item['id']}",
                }])

    if results and all(r.get("tags", {}).get("done") for r in results):
        text += "_All done! 🎉_"

    return text, {"inline_keyboard": buttons}



async def send_todo_checklist(
    chat_id: str,
    results: List[Dict],
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    results = [r for r in results if not r.get("tags", {}).get("done", False)]
    text, reply_markup = format_todo_checklist(results, date_from, date_to)
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id":      chat_id,
                "text":         text,
                "parse_mode":   "Markdown",
                "reply_markup": reply_markup,
            },
        )

        # Store the date context for this checklist so refresh knows the date
        data = resp.json()
        if data.get("ok") and date_from:
            msg_id = data["result"]["message_id"]
            async with async_session_maker() as session:
                user = await session.scalar(
                    select(User).where(User.telegram_chat_id == chat_id)
                )
                if user:
                    await context_service.set_checklist_context(
                        user.id, msg_id, date_from, date_to or date_from
                    )



async def send_list_display(chat_id: str, search_result: Dict):
    """Send a named list as an interactive Telegram checklist."""
    from services.list_service import ListService
    token     = os.getenv("TELEGRAM_BOT_TOKEN", "")
    results   = search_result.get("results", [])
    list_name = search_result.get("list_name", "List")

    # FIX: handle empty results — list doesn't exist yet
    if not results:
        await _tg_send(
            chat_id,
            search_result.get(
                "natural_response",
                f"You don't have a *{list_name}* yet.\n\n"
                f"Start by sending:\n`{list_name.lower()}:\n- item1\n- item2`"
            )
        )
        return

    list_msg = results[0].get("list_message")
    if not list_msg:
        await _tg_send(chat_id, f"📋 *{list_name}*\n\n_Nothing here yet._")
        return

    ls = ListService(cerebras_client)
    text, reply_markup = ls.format_for_telegram(list_msg)
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


async def _tg_send(chat_id: str, text: str, reply_markup: Optional[Dict] = None):
    """Thin helper for sending a plain Telegram message."""
    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(f"https://api.telegram.org/bot{token}/sendMessage", json=payload)

# ================== Message Processing ==================

TODO_SEARCH_KEYWORDS = {
    "todo", "to-do", "to do", "task", "tasks",
    "pending", "checklist", "check list",
}

# Visual content signals → auto-route to image search
IMAGE_CONTENT_SIGNALS = {
    "boarding pass", "board pass", "flight ticket", "train ticket",
    "bus ticket", "passport", "visa", "id card", "aadhar", "aadhaar",
    "pan card", "driving licence", "driving license", "voter id",
    "receipt", "invoice", "bill", "screenshot", "scan", "photo of",
    "picture of", "pic of", "image of",
}

# Phrases that strongly indicate the user wants to retrieve something
_RETRIEVAL_PHRASES = {
    "show me", "show my", "find my", "find the", "get my", "get the",
    "check my", "check the", "list my", "look up", "look for",
    "what's my", "what is my", "where's my", "where is my",
    "what did i", "what have i", "did i save", "did i add", "did i note",
    "have i saved", "have i added", "do i have", "do i still have",
    "anything about", "something about", "anything on", "something on",
    "remind me of", "remind me what", "recall my", "recall what",
    "my todo", "my todos", "my task", "my tasks", "my list", "my notes",
    "my reminders", "my shopping", "my grocery", "my groceries",
    "my events", "my ideas", "my password", "my pin", "my code",
    "my schedule", "my plans", "my appointments", "my bookings",
    "tell me about", "tell me what", "what about my",
}

# First words that (when starting a message) almost always signal retrieval
_RETRIEVAL_STARTERS = {
    "show", "find", "search", "recall", "retrieve",
    "what", "where", "when", "who", "which", "how",
    "did", "have", "is", "are", "was", "were",
}

# First words that are action/save verbs — never treat as retrieval even if
# retrieval phrases appear later (e.g. "add X to my grocery list")
_ACTION_STARTERS = {
    "add", "put", "buy", "call", "send", "pay", "book", "order",
    "create", "make", "write", "note", "save", "remind", "track",
    "log", "schedule", "set", "plan", "prepare", "draft", "email",
    "message", "text", "ping", "contact", "fix", "update", "check",
    "submit", "upload", "share", "forward", "reserve", "register",
    "transfer", "deposit", "clean", "wash", "read", "watch", "study",
}

# Overview / digest triggers
_DIGEST_TRIGGERS = {
    "overview", "digest", "summary", "summarize",
    "what do i have", "what's on my plate", "what is on my plate",
    "what's pending", "what is pending", "what's due", "what is due",
    "show everything", "show me everything", "show all",
    "what's new", "what is new", "catch me up",
}


def _is_digest_request(content: str) -> bool:
    lc = content.lower().strip().rstrip("?").strip()
    return lc in _DIGEST_TRIGGERS or any(lc.startswith(t) for t in _DIGEST_TRIGGERS)


def _is_image_query(content: str) -> bool:
    """Detect natural image retrieval without requiring 'image:' prefix."""
    lc = content.lower().strip()
    if lc.startswith("image:"):
        return True
    return any(sig in lc for sig in IMAGE_CONTENT_SIGNALS)


def _is_search_query(content: str) -> bool:
    lc = content.lower().strip()
    first = lc.split()[0] if lc.split() else ""

    # Action verbs at the start → always a save/action, never retrieval
    # even if retrieval phrases appear later ("add X to my grocery list")
    if first in _ACTION_STARTERS:
        return False

    # Legacy explicit syntax
    if lc.startswith(("search:", "find:", "get:", "image:")) or lc.endswith("?"):
        return True

    # Digest / overview
    if _is_digest_request(lc):
        return True

    # Strong retrieval phrases anywhere in the message
    if any(phrase in lc for phrase in _RETRIEVAL_PHRASES):
        return True

    # First word is a clear retrieval verb or question word
    if first in _RETRIEVAL_STARTERS:
        # Guard against false positives like "did the dishes", "have a good day"
        # Real retrieval almost always involves self-reference (my/i/me) or is a pure question
        if any(w in lc for w in (" my ", " i ", " me ", " the ", " any ")):
            return True
        # Pure question starters (what, where, when, who, which) → always retrieval
        if first in {"what", "where", "when", "who", "which"}:
            return True

    # Visual content → image retrieval
    if _is_image_query(lc):
        return True

    return False


def _extract_query(content: str) -> str:
    lc = content.lower().strip()
    if lc.startswith(("search:", "find:", "get:")):
        return content.split(":", 1)[1].strip()
    return content.strip()


async def _send_image_results(chat_id: str, results: List[Dict]):
    """Re-send saved images via Telegram sendPhoto."""
    for r in results[:3]:
        file_id = r.get("file_id")
        if not file_id:
            continue

        caption = f"🖼 *{r.get('essence', 'Image')}*"
        if r.get("caption") and r["caption"] != "[Image]":
            caption += f"\n_{r['caption']}_"
        if r.get("description"):
            caption += f"\n{r['description']}"
        caption += f"\n\n_Saved {r.get('created_at', '')[:10]}_"

        try:
            await messaging_client.send_image(
                to=chat_id,
                image_url=file_id,   # Telegram accepts file_id as photo param
                caption=caption,
            )
        except Exception as e:
            print(f"[image] Re-send failed for {file_id}: {e}")
            await messaging_client.send_message(
                chat_id, f"⚠ Couldn't retrieve image: {r.get('essence', '')}"
            )



async def process_webhook_message(webhook_data: Dict):
    # FIX: owns its own session — the request's db session is closed by the time
    # this background task runs.
    async with async_session_maker() as db:
        try:
            messages = messaging_client.extract_message_data(webhook_data)

            for msg_data in messages:
                chat_id      = msg_data["user_id"]
                content      = msg_data["content"]
                message_type = msg_data["message_type"]
                metadata     = msg_data.get("metadata", {})

                # Look up user
                if Config.get_messaging_platform() == MessagingPlatform.TELEGRAM:
                    user = await db.scalar(select(User).where(User.telegram_chat_id == chat_id))
                else:
                    user = await db.scalar(select(User).where(User.phone_number == chat_id))

                # ── /start [join_<token>] ─────────────────────────────
                if content.lower().strip() == "/start" or content.lower().startswith("/start "):
                    # Handle invite deep link: /start join_<token>
                    parts = content.split(" ", 1)
                    start_param = parts[1].strip() if len(parts) > 1 else ""
                    if start_param.startswith("join_"):
                        token = start_param[len("join_"):]
                        if user:
                            result = await grp_svc.accept_invite(token, user, db)
                            await messaging_client.send_message(chat_id, result["message"])
                        else:
                            await messaging_client.send_message(
                                chat_id,
                                "🚫 Link your account first: /link +91XXXXXXXXXX\nThen send this again."
                            )
                        continue

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
                            "👋 Welcome to Extended Brain!\n\n"
                            "1. Register at: https://your-digital-mind.vercel.app\n"
                            "2. Link Telegram: `/link +919876543210`"
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

                # ── /joingroup <token> — accept invite (works pre-login too) ───
                if content.lower().startswith("/joingroup "):
                    token     = content.split(" ", 1)[1].strip()
                    if user:
                        result = await grp_svc.accept_invite(token, user, db)
                        await messaging_client.send_message(chat_id, result["message"])
                    else:
                        await messaging_client.send_message(
                            chat_id,
                            "🚫 Link your account first: /link +91XXXXXXXXXX\nThen send this again."
                        )
                    continue

                if not user:
                    await messaging_client.send_message(
                        chat_id,
                        "🚫 Please link your account first.\n\nSend: /link +919876543210"
                    )
                    continue

                user_phone = user.phone_number

                # ── /redeem <code> ────────────────────────────────────
                if content.lower().startswith("/redeem "):
                    code   = content.split(" ", 1)[1].strip()
                    result = await cpn_svc.redeem(code, user, db)
                    if result["success"]:
                        dur = f" ({result['duration_days']} days)" if result.get("duration_days") else ""
                        await messaging_client.send_message(
                            chat_id,
                            f"🎉 *Coupon redeemed!* Pro activated{dur}\n\n"
                            f"Now try:\n`/creategroup Goa Trip`\n`/invite +91XXXXXXXXXX`"
                        )
                    else:
                        await messaging_client.send_message(chat_id, f"❌ {result['message']}")
                    continue

                # ── /upgrade ─────────────────────────────────────────
                if content.lower().strip() in {"/upgrade", "upgrade"}:
                    await messaging_client.send_message(chat_id, (
                        "⭐ *Extended Brain Pro*\n\n"
                        "Unlock collaborative groups:\n"
                        "• Add up to 6 members to your account\n"
                        "• Create unlimited groups (trip planning, family, work…)\n"
                        "• Assign tasks with @mentions\n"
                        "• Shared lists everyone can check off\n\n"
                        "💰 *₹299/month* · Cancel anytime\n\n"
                        "To get Pro, visit: https://your-digital-mind.vercel.app/#pricing\n"
                        "or contact support to activate manually."
                    ))
                    continue

                # ── /creategroup <name> ───────────────────────────────
                if content.lower().startswith("/creategroup "):
                    name   = content.split(" ", 1)[1].strip()
                    result = await grp_svc.create_group(user, name, None, db)
                    if result["success"]:
                        await messaging_client.send_message(
                            chat_id,
                            f"✅ Group *{name}* created!\n\n"
                            f"Now invite members: `/invite +91XXXXXXXXXX`\n"
                            f"Or activate it: `/setgroup {name}`"
                        )
                    else:
                        await messaging_client.send_message(chat_id, f"❌ {result['message']}")
                    continue

                # ── /invite <phone> ───────────────────────────────────
                if content.lower().startswith("/invite "):
                    phone  = content.split(" ", 1)[1].strip()
                    result = await grp_svc.invite_member(user, phone, db)
                    if result["success"]:
                        token      = result["invite_token"]
                        invitee    = result.get("invitee_name") or phone
                        invite_url = f"https://t.me/{Config.TELEGRAM_BOT_USERNAME}?start=join_{token}"
                        msg = (
                            f"✅ Invite sent to *{invitee}*!\n\n"
                            f"Share this link with them:\n{invite_url}\n\n"
                            f"Or they can send: `/joingroup {token}`"
                        )
                        # Notify invitee via Telegram if they exist
                        if result.get("invitee_exists"):
                            invitee_user = await db.scalar(select(User).where(User.phone_number == phone))
                            if invitee_user and invitee_user.telegram_chat_id:
                                await messaging_client.send_message(
                                    invitee_user.telegram_chat_id,
                                    f"👋 *{user.name}* has invited you to their Extended Brain Pro account!\n\n"
                                    f"Tap to join: `/joingroup {token}`"
                                )
                    else:
                        msg = f"❌ {result['message']}"
                    await messaging_client.send_message(chat_id, msg)
                    continue

                # ── /accountmembers ───────────────────────────────────
                if content.lower().strip() in {"/accountmembers", "/invites"}:
                    print(f"[accountmembers] triggered by user_id={user.id} phone={user.phone_number}")
                    acct = await grp_svc.get_or_create_pro_account(user, db)
                    print(f"[accountmembers] pro_account_id={acct.id} max_members={acct.max_members}")
                    members = await grp_svc.get_account_members(acct.id, db)
                    print(f"[accountmembers] found {len(members)} member(s): {[{m['phone']: m['status']} for m in members]}")
                    if not members:
                        msg = "No members in your Pro account yet.\nUse `/invite +91XXXXXXXXXX` to invite someone."
                    else:
                        lines = ["👥 *Pro Account Members*\n"]
                        for m in members:
                            status_icon = "✅" if m["status"] == "active" else "⏳"
                            lines.append(f"{status_icon} {m['name']} — {m['phone']} ({m['status']})")
                        lines.append("\nTo cancel a pending invite: `/cancelinvite +91XXXXXXXXXX`")
                        msg = "\n".join(lines)
                    print(f"[accountmembers] sending response to chat_id={chat_id}")
                    await messaging_client.send_message(chat_id, msg)
                    continue

                # ── /cancelinvite <phone> ──────────────────────────────
                if content.lower().startswith("/cancelinvite "):
                    phone = content.split(" ", 1)[1].strip()
                    result = await grp_svc.cancel_invite(user, phone, db)
                    await messaging_client.send_message(
                        chat_id, f"✅ {result['message']}" if result["success"] else f"❌ {result['message']}"
                    )
                    continue

                # ── /addmember <phone> ────────────────────────────────
                if content.lower().startswith("/addmember "):
                    phone = content.split(" ", 1)[1].strip()
                    result = await grp_svc.add_member_to_active_group(user, phone, db)
                    if result["success"]:
                        await messaging_client.send_message(
                            chat_id,
                            f"✅ *{result['name']}* added to *{result['group_name']}*!\n\n"
                            f"They can now use `/setgroup {result['group_name']}` to activate it."
                        )
                    else:
                        await messaging_client.send_message(chat_id, f"❌ {result['message']}")
                    continue

                # ── /mygroups ─────────────────────────────────────────
                if content.lower().strip() in {"/mygroups", "mygroups", "/groups"}:
                    groups = await grp_svc.get_user_groups(user.id, db)
                    response = grp_svc.format_groups_list(groups)
                    await messaging_client.send_message(chat_id, response)
                    continue

                # ── /setgroup <name> ──────────────────────────────────
                if content.lower().startswith("/setgroup "):
                    gname = content.split(" ", 1)[1].strip()
                    group = await grp_svc.get_group_by_name(user.id, gname, db)
                    if not group:
                        await messaging_client.send_message(
                            chat_id, f"❌ No group named *{gname}* found. Check `/mygroups`"
                        )
                    else:
                        await grp_svc.set_active_group(user.id, group.id, db)
                        # Reload user to get updated active_group_id
                        await db.refresh(user)
                        await messaging_client.send_message(
                            chat_id,
                            f"✅ Now posting to group *{group.name}*\n\n"
                            f"_All your messages go to the group until you send `/unsetgroup`_"
                        )
                    continue

                # ── /unsetgroup ───────────────────────────────────────
                if content.lower().strip() in {"/unsetgroup", "unsetgroup"}:
                    await grp_svc.set_active_group(user.id, None, db)
                    await messaging_client.send_message(chat_id, "✅ Back to personal mode.")
                    continue

                # ── /groupmembers ─────────────────────────────────────
                if content.lower().strip() in {"/groupmembers", "/members"}:
                    gid = user.active_group_id
                    if not gid:
                        await messaging_client.send_message(
                            chat_id, "❌ No active group. Use `/setgroup <name>` first."
                        )
                    else:
                        group = await grp_svc.get_group_by_id(gid, user.id, db)
                        if group:
                            members = await grp_svc.get_group_members(gid, db)
                            await messaging_client.send_message(
                                chat_id, grp_svc.format_group_members(group.name, members)
                            )
                    continue

                # ── /status ───────────────────────────────────────────
                if content.lower().strip() in {"/status", "status"}:
                    response = await _build_status(user, db)
                    await messaging_client.send_message(chat_id, response)
                    continue

                # ── briefing time setter ───────────────────────────────
                if content.lower().startswith("briefing:"):
                    time_str = content.split(":", 1)[1].strip()
                    if re.match(r"^\d{1,2}:\d{2}$", time_str):
                        h, m  = time_str.split(":")
                        hhmm  = f"{int(h):02d}:{int(m):02d}"
                        ok    = await briefing_service.set_briefing_time(user.id, hhmm)
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
                    # FIX: handle subtask_pick_number — user is replying with a digit
                    elif pending.get("type") == "subtask_pick_number" and lc.isdigit():
                        await _handle_subtask_pick_number(user, pending, int(lc), chat_id)
                        await context_service.clear_pending_confirmation(user.id)
                        continue

                # ── Media handling ─────────────────────────────────────
                media_url = None
                file_id   = metadata.get("file_id") or metadata.get("media_id")

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

                # ── Image message — vision processing ─────────────────────────
                if message_type == "image" and file_id:
                    try:
                        from services.vision_service import vision_service

                        # Use existing client — get URL then download bytes
                        media_url  = await messaging_client.get_media_url(file_id)
                        image_data = await messaging_client.download_media(media_url)

                        # Infer mime type from URL extension
                        ext      = media_url.rsplit(".", 1)[-1].lower()
                        mime_map = {
                            "jpg": "image/jpeg", "jpeg": "image/jpeg",
                            "png": "image/png",  "webp": "image/webp",
                            "gif": "image/gif",
                        }
                        mime_type = mime_map.get(ext, "image/jpeg")

                        ref    = datetime.utcnow()
                        result = await message_processor._process_image(
                            user=user,
                            file_id=file_id,
                            caption=content,
                            image_data=image_data,
                            mime_type=mime_type,
                            db=db,
                            ref=ref,
                        )

                        title             = result.get("essence", "Image")
                        extracted_preview = result.get("extracted_text", "")[:60]
                        extracted_line    = f"\n📄 _{extracted_preview}..._" if extracted_preview else ""

                        response = (
                            f"✓ Got it! Image saved.\n\n"
                            f"🖼 *{title}*{extracted_line}\n\n"
                            f"_Just ask me to find it anytime._"
                        )
                    except Exception as e:
                        print(f"[image] Processing failed: {e}")
                        import traceback; traceback.print_exc()
                        response = "⚠ Couldn't process the image. Please try again."

                    await messaging_client.send_message(chat_id, response)
                    continue

                # ── Audio transcription (existing) ────────────────────────────
                if message_type == "audio" and file_id:
                    try:
                        media_url = await messaging_client.get_media_url(file_id)
                        content   = await cerebras_client.transcribe_audio(media_url)
                    except Exception:
                        content = "[Audio - transcription failed]"


                try:

                    # ── IMAGE RETRIEVAL ────────────────────────────────────────────
                    # Catches both explicit "image: X" and natural queries like
                    # "show me my boarding pass" / "find that receipt"
                    if _is_image_query(content):
                        # Normalise: strip "image:" prefix if present, else use raw query
                        lc_c = content.lower().strip()
                        img_query = (
                            content.split(":", 1)[1].strip()
                            if lc_c.startswith("image:")
                            else content
                        )
                        search_result = await search_service.search(
                            user_phone=user_phone, query=f"image: {img_query}", db=db
                        )
                        results = search_result.get("results", [])
                        if not results:
                            response = search_result.get(
                                "natural_response",
                                "No images found matching that. Try describing what's in it."
                            )
                            await messaging_client.send_message(chat_id, response)
                        else:
                            await _send_image_results(chat_id, results)
                        continue


                    # ── DIGEST / OVERVIEW ─────────────────────────────
                    if _is_digest_request(content):
                        response = await _build_digest(user, db)
                        await messaging_client.send_message(chat_id, response)
                        continue

                    # ── SEARCH ────────────────────────────────────────
                    if _is_search_query(content):
                        query    = _extract_query(content)
                        prev_ctx = await context_service.get_search_context(user.id)

                        # if prev_ctx and not content.lower().startswith(("search:", "find:", "get:")):
                        #     query = f"{prev_ctx['query']} {query}"

                        search_result = await search_service.search(
                            user_phone=user_phone, query=query, db=db
                        )
                        results = search_result.get("results", [])

                        natural = search_result.get("natural_response", "")
                        await context_service.set_search_context(user.id, query, natural[:300])

                        if search_result.get("is_list") and Config.get_messaging_platform() == MessagingPlatform.TELEGRAM:
                            await send_list_display(chat_id, search_result)
                            continue

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

                        # ── Group routing: tag message with group_id + @mention ──
                        active_gid   = user.active_group_id
                        assigned_uid = None
                        if active_gid:
                            members  = await grp_svc.get_group_members(active_gid, db)
                            assigned_uid, content = grp_svc.parse_mention(content, members)

                        # ── Check list intent FIRST (before subtask) ──────────────
                        # "add to the galleria list: x, y" must route to list_service,
                        # not subtask_service — subtask detection would greedily match
                        # the list message as a parent task.
                        from services.list_service import ListService
                        ls = ListService(cerebras_client)
                        list_intent = await ls.detect_list_intent(content)

                        if list_intent and list_intent["intent"] == "create_or_add":
                            msg, added, was_created = await ls.create_or_add(
                                user.id,
                                list_intent["list_name"],
                                list_intent["list_type"],
                                list_intent["items"],
                                db,
                                group_id=active_gid,
                            )
                            action    = "Created" if was_created else "Updated"
                            list_name = list_intent["list_name"]
                            total     = len((msg.tags or {}).get("subtasks", []))

                            await messaging_client.send_message(
                                chat_id,
                                f"✓ {action} *{list_name}*  ·  {added} added · {total} total"
                            )
                            await send_list_display(chat_id, {
                                "results": [{
                                    "list_message": msg,
                                    "essence":      list_name,
                                }],
                                "is_list":   True,
                                "list_name": list_name,
                            })
                            await context_service.clear_search_context(user.id)
                            continue

                        else:

                            subtask_intent = await subtask_service.detect_subtask_intent(
                                content, user.id, db
                            )
                            if subtask_intent:
                                response = await _handle_nl_subtask(user, subtask_intent, content, chat_id, db)
                            else:
                                result = await message_processor.process(
                                    user_phone=user_phone, content=content,
                                    message_type=message_type, media_url=media_url, db=db,
                                )

                                # Tag message with group_id / assigned_to if in group mode
                                if active_gid and result.get("message_id"):
                                    await db.execute(
                                        update(Message)
                                        .where(Message.id == result["message_id"])
                                        .values(group_id=active_gid, assigned_to_user_id=assigned_uid)
                                    )
                                    await db.commit()

                                if result.get("_is_query"):
                                    qdata  = result.get("query_data", {})
                                    q_text = qdata.get("query_text", content)
                                    search_result = await search_service.search(
                                        user_phone=user_phone, query=q_text, db=db
                                    )
                                    results = search_result.get("results", [])
                                    await context_service.set_search_context(user.id, q_text, "")
                                    if search_result.get("is_list") and Config.get_messaging_platform() == MessagingPlatform.TELEGRAM:
                                        await send_list_display(chat_id, search_result)
                                        continue
                                    is_todo_q = any(kw in q_text.lower() for kw in TODO_SEARCH_KEYWORDS)
                                    if is_todo_q and Config.get_messaging_platform() == MessagingPlatform.TELEGRAM:
                                        date_from = search_result.get("date_from")
                                        date_to   = search_result.get("date_to")
                                        await send_todo_checklist(chat_id, results, date_from, date_to)
                                        continue
                                    response = format_search_results(search_result)
                                else:
                                    all_buckets = result.get("all_buckets", [])

                                    # ── List save → show full updated checklist immediately ───
                                    if result.get("is_list") and result.get("message_id"):
                                        async with async_session_maker() as session:
                                            from sqlalchemy import select as sel
                                            list_msg = await session.scalar(
                                                sel(Message).where(Message.id == result["message_id"])
                                            )
                                        if list_msg:
                                            from services.list_service import ListService
                                            ls        = ListService(cerebras_client)
                                            was_created = result.get("items_added") == result.get("total_items")
                                            action      = "Created" if was_created else "Updated"
                                            list_name   = result.get("list_name", "List")
                                            added       = result.get("items_added", 0)
                                            total       = result.get("total_items", 0)

                                            # Send header confirmation first
                                            await messaging_client.send_message(
                                                chat_id,
                                                f"✓ {action} *{list_name}*  ·  {added} added · {total} total"
                                            )
                                            # Then send full interactive checklist
                                            await send_list_display(chat_id, {
                                                "results": [{
                                                    "list_message": list_msg,
                                                    "essence":      list_name,
                                                }],
                                                "is_list":   True,
                                                "list_name": list_name,
                                            })
                                            await context_service.clear_search_context(user.id)
                                            continue

                                    # ── All other saves ───────────────────────────────────────
                                    response = _build_save_response(result)
                                    skip_buckets = {"Track", "Random"}
                                    if not result.get("is_list") and not set(all_buckets) <= skip_buckets:
                                        asyncio.create_task(
                                            _check_project(user, result, content, chat_id)
                                        )
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

async def _build_digest(user: User, db: AsyncSession) -> str:
    """Quick overview: today's todos, upcoming events, and recent saves."""
    from datetime import date as date_type
    today     = str(date_type.today())
    tomorrow  = str(date_type.today() + timedelta(days=1))
    week_end  = str(date_type.today() + timedelta(days=7))

    # Today's pending todos
    todo_res = await db.execute(
        select(Message.summary, Message.tags)
        .where(
            Message.user_id == user.id,
            text("messages.tags->>'due_date' = :today"),
            text("(messages.tags->>'done')::boolean IS NOT TRUE"),
            text("messages.tags->'all_buckets' @> '\"To-Do\"'::jsonb"),
        )
        .params(today=today)
        .order_by(Message.created_at.asc())
        .limit(5)
    )
    todos = todo_res.all()

    # Upcoming events (next 7 days)
    events_res = await db.execute(
        select(Message.summary, Message.tags)
        .where(
            Message.user_id == user.id,
            text("messages.tags->>'due_date' >= :today"),
            text("messages.tags->>'due_date' <= :week_end"),
            text("messages.tags->'all_buckets' @> '\"Events\"'::jsonb"),
        )
        .params(today=today, week_end=week_end)
        .order_by(text("messages.tags->>'due_date' ASC"))
        .limit(3)
    )
    events = events_res.all()

    # Last 3 saves (any type)
    recent_res = await db.execute(
        select(Message.summary, Message.tags)
        .where(Message.user_id == user.id)
        .order_by(Message.created_at.desc())
        .limit(3)
    )
    recent = recent_res.all()

    lines = [f"📋 *Here's what's on your plate, {user.name.split()[0]}*\n"]

    if todos:
        lines.append("*Today's tasks:*")
        for summary, tags in todos:
            t = tags or {}
            evt_time = t.get("event_time", "")
            time_str = f" _{evt_time}_" if evt_time else ""
            lines.append(f"• {summary or 'task'}{time_str}")
    else:
        lines.append("✅ No tasks due today.")

    if events:
        lines.append("\n*Upcoming events:*")
        for summary, tags in events:
            t = tags or {}
            due = t.get("due_date", "")
            evt_time = t.get("event_time", "")
            when = f"{due} {evt_time}".strip()
            lines.append(f"• {summary or 'event'} _{when}_")

    if recent:
        lines.append("\n*Recently saved:*")
        for summary, tags in recent:
            t = tags or {}
            buckets = t.get("all_buckets", ["Note"])
            label = buckets[0] if buckets else "Note"
            lines.append(f"• [{label}] {summary or '…'}")

    return "\n".join(lines)


async def _build_status(user: User, db: AsyncSession) -> str:
    today      = datetime.utcnow().strftime("%Y-%m-%d")
    week_start = (datetime.utcnow() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")
    week_end   = (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d")

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

    saves_res = await db.execute(
        select(func.count(Message.id))
        .where(Message.user_id == user.id, Message.created_at >= week_start)
    )
    saves_week = saves_res.scalar() or 0

    ideas_res = await db.execute(
        select(func.count(Message.id))
        .where(
            Message.user_id == user.id,
            text("messages.tags->'all_buckets' @> '\"Ideas\"'::jsonb"),
        )
    )
    ideas = ideas_res.scalar() or 0

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

    rec_res = await db.execute(
        select(func.count())
        .select_from(text("recurrences"))
        .where(text("user_id = :uid AND is_active = TRUE"))
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
            async with async_session_maker() as session:
                msg = await session.scalar(select(Message).where(Message.id == candidates[0].id))
            if msg:
                text_, reply_markup = subtask_service.format_subtasks(msg)
                await _tg_send(chat_id, text_, reply_markup)
            return ""
        return "❌ Failed to add subtasks."

    options = "\n".join(f"{i+1}. {c.content[:60]}" for i, c in enumerate(candidates[:5]))
    await context_service.set_pending_confirmation(user.id, "subtask_pick", {
        "candidates": [{"id": c.id, "content": c.content} for c in candidates[:5]],
        "subtasks":   subs,
    })
    return (
        f"🤔 Found {len(candidates)} matching tasks:\n\n{options}\n\n"
        f"Which one? Reply with the number (1-{min(len(candidates), 5)})"
    )


async def _handle_subtask_pick_number(
    user: User, pending: Dict, number: int, chat_id: str
):
    """FIX: was missing entirely — handles user replying with a digit after subtask_pick."""
    data       = pending.get("data", {})
    candidates = data.get("candidates", [])
    subs       = data.get("subtasks", [])
    idx        = number - 1

    if idx < 0 or idx >= len(candidates):
        await messaging_client.send_message(
            chat_id, f"❌ Please reply with a number between 1 and {len(candidates)}."
        )
        return

    chosen_id = candidates[idx]["id"]
    ok = await subtask_service.add_subtasks(chosen_id, subs)
    if ok:
        async with async_session_maker() as session:
            msg = await session.scalar(select(Message).where(Message.id == chosen_id))
        if msg:
            text_, reply_markup = subtask_service.format_subtasks(msg)
            await _tg_send(chat_id, text_, reply_markup)
            return
    await messaging_client.send_message(chat_id, "❌ Failed to add subtasks.")


async def _handle_nl_subtask(
    user: User, intent: Dict, content: str, chat_id: str, db: AsyncSession
) -> str:
    parent_id = intent.get("parent_id")
    subs      = intent.get("subtasks", [])
    hint      = intent.get("parent_hint", "")

    if parent_id:
        # Check if the matched parent is actually a List message
        # If so, use list_add_confirm instead of subtask_confirm
        async with async_session_maker() as session:
            from sqlalchemy import select as sel
            parent_msg = await session.scalar(sel(Message).where(Message.id == parent_id))

        if parent_msg:
            parent_tags    = parent_msg.tags if isinstance(parent_msg.tags, dict) else {}
            parent_buckets = parent_tags.get("all_buckets", [])

            if parent_tags.get("is_list"):
                # This is a named list — use list_add_confirm
                list_name = parent_tags.get("list_name", parent_msg.content)
                list_type = parent_tags.get("list_type", "custom")
                await context_service.set_pending_confirmation(user.id, "list_add_confirm", {
                    "list_name": list_name,
                    "list_type": list_type,
                    "items":     subs,
                })
                return (
                    f"🛒 Add these items to *{list_name}*?\n\n"
                    + "\n".join(f"• {s}" for s in subs)
                    + "\n\nReply *yes* to confirm or *no* to skip."
                )

        # Original subtask flow for non-list parents
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
        candidates = await subtask_service.find_parent_message(user.id, hint, db)
        if not candidates:
            return f"❌ Couldn't find a task matching *{hint}*."
        await context_service.set_pending_confirmation(user.id, "subtask_pick", {
            "candidates": [{"id": c.id, "content": c.content} for c in candidates[:5]],
            "subtasks":   subs,
        })
        options = "\n".join(f"{i+1}. {c.content[:60]}" for i, c in enumerate(candidates[:5]))
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


async def _check_project(user: User, result: Dict, content: str, chat_id: str):
    """Background task — detect project. Opens its own session."""
    try:
        async with async_session_maker() as session:
            msg = await session.scalar(
                select(Message).where(Message.id == result.get("message_id"))
            )
            if not msg:
                return

            analysis = {
                "concepts": result.get("connections", []),
                "keywords": result.get("tags", []),
            }

            project = await project_service.detect_and_suggest(user, msg, analysis, session)
            if not project:
                return

        await context_service.set_pending_confirmation(user.id, "project_confirm", {
            "message_id": msg.id,
            "project":    project,
        })

        await _tg_send(
            chat_id,
            f"📁 Add this to project *{project}*?",
            reply_markup={
                "inline_keyboard": [[
                    {"text": "✅ Yes", "callback_data": f"proj_yes:{msg.id}:{project}"},
                    {"text": "❌ No",  "callback_data": f"proj_no:{msg.id}"},
                ]]
            },
        )
    except Exception as e:
        print(f"[project] Background check failed: {e}")


async def _handle_confirmation_yes(
    user: User, pending: Dict, chat_id: str, db: AsyncSession
):
    conf_type = pending.get("type")
    data      = pending.get("data", {})

        # NEW: list add confirmation
    if conf_type == "list_add_confirm":
        from services.list_service import ListService
        ls = ListService(cerebras_client)
        msg, added, _ = await ls.create_or_add(
            user.id,
            data["list_name"],
            data["list_type"],
            data["items"],
            db,
        )
        text_, reply_markup = ls.format_for_telegram(msg)
        await _tg_send(chat_id, text_, reply_markup)
        return

    elif conf_type == "subtask_confirm":
        ok = await subtask_service.add_subtasks(data["message_id"], data["subtasks"])
        if ok:
            async with async_session_maker() as session:
                msg = await session.scalar(select(Message).where(Message.id == data["message_id"]))
            if msg:
                text_, reply_markup = subtask_service.format_subtasks(msg)
                await _tg_send(chat_id, text_, reply_markup)
                return
        await messaging_client.send_message(chat_id, "❌ Failed to add subtasks.")

    elif conf_type == "subtask_pick":
        # User said "yes" to an ambiguous pick — prompt for the number
        await messaging_client.send_message(chat_id, "Reply with the number of the task (e.g. `1`)")
        await context_service.set_pending_confirmation(user.id, "subtask_pick_number", data)

    elif conf_type == "project_confirm":
        await project_service.assign_project(data["message_id"], data["project"])
        await messaging_client.send_message(chat_id, f"📁 Added to project *{data['project']}*!")


# ================== Helpers ==================
def _build_save_response(result: Dict) -> str:
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
        return f"✓ Got it!\n\n📝 {essence}\n⏰ I'll remind you at {time_str} IST."

    if split_count > 1:
        return f"✓ Got it! Saved {split_count} tasks.\n\n📝 {essence}"

    if "To-Do" in all_buckets and due_date:
        return f"✓ Added to your to-do list!\n\n📝 {essence}\n📅 {due_date}"

    if "Ideas" in all_buckets:
        return f"✓ Idea saved!\n\n💡 {essence}"

    if "Track" in all_buckets:
        return f"✓ Logged!\n\n📊 {essence}"

    if "Events" in all_buckets:
        return f"✓ Event saved!\n\n📅 {essence}"

    return f"✓ Got it!\n\n📝 {essence}"



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


# ================== REST API Endpoints ==================

@app.get("/api/messages/assigned")
async def get_assigned_messages(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return tasks assigned to the current user across all groups.

    Two cases:
    1. Explicitly assigned (@mention): assigned_to_user_id == current_user.id
    2. Group-wide unassigned To-Do: no specific assignee yet, user is a group member
       (not the sender). Once retroactively assigned, case 2 exits naturally because
       assigned_to_user_id becomes non-null and tags gains an 'assignments' key.
    """
    _user_group_ids = select(GroupMember.group_id).where(GroupMember.user_id == current_user.id)
    result = await db.execute(
        select(Message, Category, User)
        .outerjoin(Category, Message.category_id == Category.id)
        .join(User, User.id == Message.user_id)
        .where(
            or_(
                # Case 1: explicitly assigned to me (original behaviour)
                and_(
                    Message.assigned_to_user_id == current_user.id,
                    Message.user_id != current_user.id,   # exclude mirror messages
                ),
                # Case 2: group-wide unassigned To-Do where I'm a member
                # (includes tasks I posted myself — sender also owns responsibility)
                and_(
                    Message.group_id.in_(_user_group_ids),
                    Message.assigned_to_user_id.is_(None),
                    text("NOT (messages.tags ? 'assignments')"),
                    or_(
                        text("messages.tags->>'primary_bucket' = 'To-Do'"),
                        text("messages.tags->>'intent_bucket' = 'To-Do'"),
                    ),
                    text("COALESCE((messages.tags->>'done')::boolean, false) = false"),
                ),
            )
        )
        .order_by(Message.created_at.desc())
        .limit(100)
    )
    rows = result.all()
    messages = []
    for msg, cat, sender in rows:
        tags = msg.tags if isinstance(msg.tags, dict) else {}
        bucket = tags.get("primary_bucket") or tags.get("intent_bucket") or (cat.name if cat else "To-Do")
        raw_items = tags.get("subtasks", [])
        items = [{"task": s["task"], "done": s.get("done", False)}
                 for s in raw_items if isinstance(s, dict) and "task" in s]
        messages.append({
            "id":             msg.id,
            "content":        msg.content,
            "essence":        msg.summary or msg.content[:80],
            "message_type":   msg.message_type.value if msg.message_type else "text",
            "media_url":      msg.media_url,
            "category":       bucket,
            "all_buckets":    tags.get("all_buckets", [bucket]),
            "priority":       tags.get("priority", "normal"),
            "tags":           tags,
            "created_at":     msg.created_at.isoformat(),
            "due_date":       tags.get("due_date"),
            "is_done":        bool(tags.get("done", False)),
            "is_list":        bool(tags.get("is_list", False)),
            "items":          items,
            "group_id":       msg.group_id,
            "sender_name":    sender.name,
            "sender_id":      msg.user_id,
            "assigned_to_user_id": msg.assigned_to_user_id,
            "assignments":    tags.get("assignments", []),
            "expense_amount":   tags.get("expense_amount"),
            "expense_category": tags.get("expense_category"),
            "expense_payer_id":   tags.get("expense_payer_id"),
            "expense_payer_name": tags.get("expense_payer_name"),
        })
    return {"success": True, "results": messages, "total": len(messages)}


@app.get("/api/messages/assigned-to-others")
async def get_assigned_to_others(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return group messages where the current user is the assigner (sender) and
    tags.assignments is non-empty — i.e. tasks they assigned to other group members."""
    import json as _json

    rows = await db.execute(
        select(Message)
        .where(
            Message.user_id == current_user.id,
            Message.group_id.isnot(None),
            text("messages.tags ? 'assignments'"),
            text("jsonb_array_length(messages.tags->'assignments') > 0"),
        )
        .order_by(Message.created_at.desc())
        .limit(200)
    )
    messages = []
    for (msg,) in rows.all():
        tags = msg.tags if isinstance(msg.tags, dict) else {}
        bucket = tags.get("primary_bucket") or tags.get("intent_bucket") or "To-Do"
        messages.append({
            "id":           msg.id,
            "content":      msg.content,
            "essence":      msg.summary or msg.content[:80],
            "message_type": msg.message_type.value if msg.message_type else "text",
            "category":     bucket,
            "tags":         tags,
            "due_date":     tags.get("due_date"),
            "created_at":   msg.created_at.isoformat(),
            "group_id":     msg.group_id,
            "assignments":  tags.get("assignments", []),
        })
    return {"success": True, "results": messages, "total": len(messages)}


@app.patch("/api/messages/{message_id}/assignments/{assignment_idx}/complete")
async def complete_assignment(
    message_id: int,
    assignment_idx: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Mark a specific assignee's slot as done.
    Callable by the assignee themselves. Notifies the assigner (message owner) via APNs."""
    import json as _json

    msg = await db.get(Message, message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")

    tags = msg.tags if isinstance(msg.tags, dict) else {}
    assignments: list[dict] = tags.get("assignments", [])

    if assignment_idx < 0 or assignment_idx >= len(assignments):
        raise HTTPException(status_code=400, detail="Invalid assignment index")

    slot = assignments[assignment_idx]
    if slot.get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="You are not the assignee for this slot")

    slot["done"]    = True
    slot["done_at"] = datetime.utcnow().isoformat()
    assignments[assignment_idx] = slot

    await db.execute(
        text(
            "UPDATE messages SET tags = tags || CAST(:extra AS jsonb) WHERE id = :mid"
        ).bindparams(
            extra=_json.dumps({"assignments": assignments}),
            mid=message_id,
        )
    )
    await db.commit()

    # ── Broadcast to group so other clients refresh ───────────────────
    if msg.group_id:
        asyncio.create_task(ws_manager.broadcast(msg.group_id, {
            "type":           "assignment_complete",
            "message_id":     message_id,
            "assignment_idx": assignment_idx,
            "completed_by":   current_user.name,
        }))

    # ── Notify the assigner (message owner) ──────────────────────────
    if msg.user_id != current_user.id:
        try:
            tokens_rows = await db.execute(
                select(DeviceToken.token).where(DeviceToken.user_id == msg.user_id)
            )
            task_preview = (msg.summary or msg.content)[:60]
            badge = await total_unread_for_user(db, msg.user_id)
            for (token,) in tokens_rows.all():
                await send_apns_notification(
                    device_token=token,
                    title=f"✅ {current_user.name} completed a task",
                    body=task_preview,
                    badge=badge,
                    data={"type": "assignment_complete", "message_id": message_id,
                          "group_id": msg.group_id},
                )
        except Exception as e:
            print(f"[push] assigner completion notify failed: {e}")

    return {
        "success":     True,
        "message_id":  message_id,
        "assignment":  slot,
        "assignments": assignments,
    }


@app.patch("/api/messages/{message_id}/assign")
async def assign_message(
    message_id: int,
    payload: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retroactively assign an existing group message to a member.
    Callable by the message owner. Appends to tags.assignments, forces bucket=To-Do,
    mirrors a personal To-Do for the assignee, and sends APNs."""
    import json as _json

    msg = await db.get(Message, message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    if msg.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Only the message owner can assign it")
    if not msg.group_id:
        raise HTTPException(status_code=400, detail="Message is not in a group")

    assignee_id   = int(payload.get("user_id", 0))
    assignee_name = str(payload.get("name", ""))
    assignee_phone = str(payload.get("phone", ""))
    if not assignee_id or not assignee_name:
        raise HTTPException(status_code=422, detail="user_id and name are required")
    if assignee_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot assign a task to yourself")

    tags: dict = msg.tags if isinstance(msg.tags, dict) else {}
    assignments: list[dict] = list(tags.get("assignments", []))

    # Prevent duplicate slot for the same assignee
    if any(a.get("user_id") == assignee_id for a in assignments):
        raise HTTPException(status_code=409, detail="Already assigned to this member")

    new_slot = {
        "user_id": assignee_id,
        "name":    assignee_name,
        "phone":   assignee_phone,
        "done":    False,
        "done_at": None,
    }
    assignments.append(new_slot)

    # Update tags AND assigned_to_user_id column so GET /api/messages/assigned works for the assignee.
    # assigned_to_user_id tracks the primary (most recent) assignee for the single-assignee query path.
    await db.execute(
        text(
            "UPDATE messages "
            "SET tags = tags || CAST(:extra AS jsonb), "
            "    assigned_to_user_id = :auid "
            "WHERE id = :mid"
        ).bindparams(
            extra=_json.dumps({
                "assignments":    assignments,
                "primary_bucket": "To-Do",
                "intent_bucket":  "To-Do",
            }),
            auid=assignee_id,
            mid=message_id,
        )
    )

    # Mirror a personal To-Do in the assignee's feed (same as capture path)
    content_preview = msg.summary or msg.content
    todo_tags = {
        "primary_bucket":    "To-Do",
        "intent_bucket":     "To-Do",
        "assigned_by":       current_user.name,
        "assigned_by_id":    current_user.id,
        "group_id":          msg.group_id,
        "source_message_id": message_id,
    }
    mirror = Message(
        user_id=assignee_id,
        content=msg.content,
        message_type=msg.message_type,
        summary=content_preview[:80],
        tags=todo_tags,
        assigned_to_user_id=assignee_id,
    )
    db.add(mirror)
    await db.flush()

    await db.commit()

    # Push notification to assignee
    try:
        tokens_rows = await db.execute(
            select(DeviceToken.token).where(DeviceToken.user_id == assignee_id)
        )
        badge = await total_unread_for_user(db, assignee_id)
        for (token,) in tokens_rows.all():
            await send_apns_notification(
                device_token=token,
                title=f"{current_user.name} assigned you a task",
                body=content_preview[:80],
                badge=badge,
                data={"type": "assignment", "group_id": msg.group_id,
                      "message_id": message_id},
            )
    except Exception as e:
        print(f"[push] assign notify failed for uid={assignee_id}: {e}")

    # Broadcast to group so other clients refresh
    asyncio.create_task(ws_manager.broadcast(msg.group_id, {
        "type":           "assignment_added",
        "message_id":     message_id,
        "assigned_to":    assignee_name,
        "assigned_to_id": assignee_id,
        "assigned_by":    current_user.name,
    }))

    # Bust personal bootstrap cache for every group member so the task transitions
    # from "group-wide unassigned" to "assigned to [name]" without waiting for TTL.
    from services import redis_cache as _rc
    try:
        member_rows = await db.execute(
            select(GroupMember.user_id).where(GroupMember.group_id == msg.group_id)
        )
        for (muid,) in member_rows.all():
            asyncio.create_task(_rc.cache_del(_rc.bootstrap_key(muid, None)))
    except Exception:
        pass  # cache bust is best-effort

    return {
        "success":     True,
        "message_id":  message_id,
        "assignment":  new_slot,
        "assignments": assignments,
    }


@app.get("/api/bootstrap")
async def bootstrap(
    group_id: Optional[int] = None,
    limit: int = 100,
    refresh: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Single round-trip that returns everything iOS needs to open a context:
    recent messages, group members, assigned tasks, and unread counts.
    Pass refresh=true to bypass the 30-second Redis cache (e.g. pull-to-refresh)."""
    from services import redis_cache
    bk = redis_cache.bootstrap_key(current_user.id, group_id)
    if not refresh:
        cached = await redis_cache.cache_get(bk)
        if cached:
            return cached

    # 1. Recent messages ──────────────────────────────────────────────
    if group_id:
        group = await grp_svc.get_group_by_id(group_id, current_user.id, db)
        if not group:
            raise HTTPException(status_code=403, detail="Not a member of this group")
        base_filter = Message.group_id == group_id
    else:
        # Exclude mirror messages (assigned_to_user_id IS NOT NULL, group_id NULL) —
        # those are personal Todo copies created for group @mention assignments and
        # belong in the assigned feed only, not in the personal dump chat.
        base_filter = and_(
            Message.user_id == current_user.id,
            Message.group_id.is_(None),
            Message.assigned_to_user_id.is_(None),
        )

    recent_rows = await db.execute(
        select(Message, Category, User)
        .outerjoin(Category, Message.category_id == Category.id)
        .outerjoin(User, Message.user_id == User.id)
        .where(base_filter)
        .order_by(Message.created_at.desc())
        .limit(min(limit, 200))
    )

    def _serialize(msg, cat, sender):
        tags = msg.tags if isinstance(msg.tags, dict) else {}
        bucket = tags.get("primary_bucket") or tags.get("intent_bucket") or (cat.name if cat else "Random")
        raw_items = tags.get("subtasks", [])
        items = [{"task": s["task"], "done": s.get("done", False)}
                 for s in raw_items if isinstance(s, dict) and "task" in s]
        return {
            "id":                  msg.id,
            "content":             msg.content,
            "essence":             msg.summary or msg.content[:80],
            "message_type":        msg.message_type.value if msg.message_type else "text",
            "media_url":           msg.media_url,
            "category":            bucket,
            "all_buckets":         tags.get("all_buckets", [bucket]),
            "priority":            tags.get("priority", "normal"),
            "tags":                tags,
            "created_at":          msg.created_at.isoformat(),
            "due_date":            tags.get("due_date"),
            "is_done":             bool(tags.get("done", False)),
            "is_list":             bool(tags.get("is_list", False)),
            "items":               items,
            "event_time":          tags.get("event_time"),
            "events":              tags.get("events", []),
            "starred":             tags.get("starred", False),
            "group_id":            msg.group_id,
            "assigned_to_user_id": msg.assigned_to_user_id,
            "sender_name":         sender.name if sender else None,
            "sender_id":           msg.user_id,
            "expense_amount":      tags.get("expense_amount"),
            "expense_category":    tags.get("expense_category"),
            "expense_payer_id":    tags.get("expense_payer_id"),
            "expense_payer_name":  tags.get("expense_payer_name"),
            "assignments":         tags.get("assignments", []),
        }

    recent = [_serialize(m, c, s) for m, c, s in recent_rows.all()]

    # 2. Group members ────────────────────────────────────────────────
    members: list = []
    if group_id:
        members = await grp_svc.get_group_members(group_id, db)

    # 3. Assigned tasks ───────────────────────────────────────────────
    # Same dual-condition as GET /api/messages/assigned (keep in sync):
    # Case 1 — explicitly assigned (@mention); Case 2 — group-wide unassigned To-Do.
    _bstrap_group_ids = select(GroupMember.group_id).where(GroupMember.user_id == current_user.id)
    assigned_rows = await db.execute(
        select(Message, Category, User)
        .outerjoin(Category, Message.category_id == Category.id)
        .join(User, User.id == Message.user_id)
        .where(
            or_(
                and_(
                    Message.assigned_to_user_id == current_user.id,
                    Message.user_id != current_user.id,
                ),
                and_(
                    Message.group_id.in_(_bstrap_group_ids),
                    # sender is included — they own the responsibility too
                    Message.assigned_to_user_id.is_(None),
                    text("NOT (messages.tags ? 'assignments')"),
                    or_(
                        text("messages.tags->>'primary_bucket' = 'To-Do'"),
                        text("messages.tags->>'intent_bucket' = 'To-Do'"),
                    ),
                    text("COALESCE((messages.tags->>'done')::boolean, false) = false"),
                ),
            )
        )
        .order_by(Message.created_at.desc())
        .limit(100)
    )
    assigned = []
    for msg, cat, sender in assigned_rows.all():
        tags = msg.tags if isinstance(msg.tags, dict) else {}
        bucket = tags.get("primary_bucket") or tags.get("intent_bucket") or (cat.name if cat else "To-Do")
        raw_items = tags.get("subtasks", [])
        items = [{"task": s["task"], "done": s.get("done", False)}
                 for s in raw_items if isinstance(s, dict) and "task" in s]
        assigned.append({
            "id":                  msg.id,
            "content":             msg.content,
            "essence":             msg.summary or msg.content[:80],
            "message_type":        msg.message_type.value if msg.message_type else "text",
            "media_url":           msg.media_url,
            "category":            bucket,
            "all_buckets":         tags.get("all_buckets", [bucket]),
            "priority":            tags.get("priority", "normal"),
            "tags":                tags,
            "created_at":          msg.created_at.isoformat(),
            "due_date":            tags.get("due_date"),
            "is_done":             bool(tags.get("done", False)),
            "is_list":             bool(tags.get("is_list", False)),
            "items":               items,
            "group_id":            msg.group_id,
            "sender_name":         sender.name,
            "assigned_to_user_id": msg.assigned_to_user_id,
            "sender_id":           msg.user_id,
            "assignments":         tags.get("assignments", []),
        })

    # 4. Unread counts ────────────────────────────────────────────────
    unread_counts: dict[str, int] = {}
    memberships = await db.execute(
        select(GroupMember.group_id).where(GroupMember.user_id == current_user.id)
    )
    gids = [r[0] for r in memberships.all()]
    if gids:
        seen_rows = await db.execute(
            select(GroupLastSeen).where(
                GroupLastSeen.user_id == current_user.id,
                GroupLastSeen.group_id.in_(gids),
            )
        )
        last_seen = {r.group_id: r.last_seen_at for r in seen_rows.scalars()}
        for gid in gids:
            cutoff = last_seen.get(gid, datetime.min)
            cnt = await db.scalar(
                select(func.count(Message.id)).where(
                    Message.group_id == gid,
                    Message.created_at > cutoff,
                    Message.user_id != current_user.id,
                )
            )
            unread_counts[str(gid)] = cnt or 0

    payload = {
        "success":       True,
        "recent":        recent,
        "members":       members,
        "assigned":      assigned,
        "unread_counts": unread_counts,
    }
    await redis_cache.cache_set(bk, payload, ex=30)
    return payload


@app.get("/api/messages/recent")
async def get_recent_messages(
    limit: int = 100,
    group_id: Optional[int] = None,
    after: Optional[str] = None,   # ISO timestamp — only return messages newer than this
    is_list: Optional[bool] = None,  # when true, return only messages with is_list=true in tags
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return recent messages — personal if no group_id, group feed if group_id provided.
    Pass ?after=<iso_ts> to get only messages newer than that timestamp (for polling).
    Pass ?is_list=true to return only named-list messages regardless of bucket."""
    if group_id:
        group = await grp_svc.get_group_by_id(group_id, current_user.id, db)
        if not group:
            raise HTTPException(status_code=403, detail="Not a member of this group")
        base_filter = Message.group_id == group_id
    else:
        base_filter = and_(
            Message.user_id == current_user.id,
            Message.group_id.is_(None),
            Message.assigned_to_user_id.is_(None),
        )

    # Parse optional `after` timestamp for incremental polling
    after_dt = None
    if after:
        try:
            after_dt = datetime.fromisoformat(after.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            pass

    query = (
        select(Message, Category, User)
        .outerjoin(Category, Message.category_id == Category.id)
        .outerjoin(User, Message.user_id == User.id)
        .where(base_filter)
    )
    if after_dt:
        query = query.where(Message.created_at > after_dt)
    if is_list:
        query = query.where(text("(messages.tags->>'is_list')::boolean = true"))

    result = await db.execute(
        query.order_by(Message.created_at.desc()).limit(min(limit, 200))
    )
    rows = result.all()

    messages = []
    for msg, cat, sender in rows:
        tags = msg.tags if isinstance(msg.tags, dict) else {}
        bucket = (
            tags.get("primary_bucket")
            or tags.get("intent_bucket")
            or (cat.name if cat else "Random")
        )
        due_date_val = tags.get("due_date")
        raw_items = tags.get("subtasks", [])
        items = [{"task": s["task"], "done": s.get("done", False)}
                 for s in raw_items if isinstance(s, dict) and "task" in s]
        messages.append({
            "id":                    msg.id,
            "content":               msg.content,
            "essence":               msg.summary or msg.content[:80],
            "message_type":          msg.message_type.value if msg.message_type else "text",
            "media_url":             msg.media_url,
            "category":              bucket,
            "all_buckets":           tags.get("all_buckets", [bucket]),
            "priority":              tags.get("priority", "normal"),
            "tags":                  tags,
            "created_at":            msg.created_at.isoformat(),
            "due_date":              due_date_val,
            "is_done":               bool(tags.get("done", False)),
            "is_list":               bool(tags.get("is_list", False)),
            "settlement_marker":     bool(tags.get("settlement_marker", False)),
            "items":                 items,
            "event_time":            tags.get("event_time"),
            "events":                tags.get("events", []),
            "starred":               tags.get("starred", False),
            "group_id":              msg.group_id,
            "assigned_to_user_id":   msg.assigned_to_user_id,
            "sender_name":           sender.name if sender else None,
            "sender_id":             msg.user_id,
            "expense_amount":        tags.get("expense_amount"),
            "expense_category":      tags.get("expense_category"),
            "expense_payer_id":      tags.get("expense_payer_id"),
            "expense_payer_name":    tags.get("expense_payer_name"),
            "assignments":           tags.get("assignments", []),
        })

    return {"success": True, "results": messages, "total": len(messages)}


# ── Group unread counts ────────────────────────────────────────────────────────

@app.get("/api/groups/unread")
async def get_unread_counts(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return unread message counts per group for the current user."""
    # Get user's groups
    memberships = await db.execute(
        select(GroupMember.group_id).where(GroupMember.user_id == current_user.id)
    )
    group_ids = [r[0] for r in memberships.all()]
    if not group_ids:
        return {"counts": {}}

    # Get last-seen timestamps
    seen_rows = await db.execute(
        select(GroupLastSeen).where(
            GroupLastSeen.user_id == current_user.id,
            GroupLastSeen.group_id.in_(group_ids)
        )
    )
    last_seen: dict[int, datetime] = {r.group_id: r.last_seen_at for r in seen_rows.scalars()}

    counts = {}
    for gid in group_ids:
        cutoff = last_seen.get(gid, datetime.min)
        count = await db.scalar(
            select(func.count(Message.id)).where(
                Message.group_id == gid,
                Message.created_at > cutoff,
                Message.user_id != current_user.id,   # don't count own messages
            )
        )
        counts[gid] = count or 0

    return {"counts": counts}


@app.post("/api/groups/{group_id}/seen")
async def mark_group_seen(
    group_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Mark all current messages in a group as seen by this user."""
    existing = await db.scalar(
        select(GroupLastSeen).where(
            GroupLastSeen.user_id == current_user.id,
            GroupLastSeen.group_id == group_id
        )
    )
    now = datetime.utcnow()
    if existing:
        existing.last_seen_at = now
    else:
        db.add(GroupLastSeen(user_id=current_user.id, group_id=group_id, last_seen_at=now))
    await db.commit()
    return {"success": True}


@app.websocket("/ws/group/{group_id}")
async def group_websocket(
    websocket: WebSocket,
    group_id: int,
    token: str = Query(...),
):
    """WebSocket endpoint for live group chat. Eliminates HTTP polling."""
    from jose import jwt as jose_jwt, JWTError
    SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key")

    # Verify JWT token from query param
    try:
        payload  = jose_jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id  = int(payload.get("sub", 0))
    except (JWTError, ValueError, TypeError):
        await websocket.close(code=1008, reason="Unauthorized")
        return

    async with async_session_maker() as db:
        user  = await db.get(User, user_id)
        if not user:
            await websocket.close(code=1008, reason="User not found")
            return
        group = await grp_svc.get_group_by_id(group_id, user_id, db)
        if not group:
            await websocket.close(code=1008, reason="Not a member")
            return

    await ws_manager.connect(websocket, group_id, user_id)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=25)
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send server-initiated keepalive; if send fails the socket is dead
                try:
                    await websocket.send_text("ping")
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket, group_id, user_id)


@app.delete("/api/messages/{message_id}")
async def delete_message(
    message_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a message owned by the current user."""
    msg = await db.get(Message, message_id)
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")
    if msg.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your message")
    await db.execute(sql_delete(Reminder).where(Reminder.message_id == message_id))
    await db.execute(update(LabelAnnotation).where(LabelAnnotation.message_id == message_id).values(message_id=None))
    await db.delete(msg)
    await db.commit()
    return {"success": True}


@app.get("/api/reminders")
async def list_reminders(
    group_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return non-cancelled reminders scoped to personal brain or a specific group."""
    stmt = (
        select(Reminder)
        .outerjoin(Message, Reminder.message_id == Message.id)
        .where(Reminder.user_id == current_user.id, Reminder.is_cancelled == False)
    )
    if group_id is not None:
        stmt = stmt.where(Message.group_id == group_id)
    else:
        stmt = stmt.where(or_(Reminder.message_id == None, Message.group_id == None))
    stmt = stmt.order_by(Reminder.remind_at.desc()).limit(200)
    result = await db.execute(stmt)
    reminders = result.scalars().all()
    now = datetime.utcnow()
    items = []
    for r in reminders:
        items.append({
            "id":           r.id,
            "task":         r.task,
            "content":      r.content,
            "remind_at":    r.remind_at.isoformat(),
            "is_sent":      r.is_sent,
            "is_past":      r.remind_at < now,
            "snooze_count": r.snooze_count,
            "message_id":   r.message_id,
        })
    return {"success": True, "reminders": items}


@app.patch("/api/reminders/{reminder_id}/snooze")
async def snooze_reminder(
    reminder_id: int,
    minutes: int = 60,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Snooze a reminder by the given number of minutes (default 60)."""
    reminder = await db.get(Reminder, reminder_id)
    if not reminder or reminder.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Reminder not found")
    snoozed = await reminder_service.snooze(reminder_id, minutes, db)
    if not snoozed:
        raise HTTPException(status_code=500, detail="Snooze failed")
    return {"success": True, "remind_at": snoozed.remind_at.isoformat()}


@app.delete("/api/reminders/fired")
async def clear_fired_reminders(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Bulk-cancel all already-sent reminders so they no longer appear in the list."""
    await db.execute(
        update(Reminder)
        .where(
            Reminder.user_id == current_user.id,
            Reminder.is_sent == True,
            Reminder.is_cancelled == False,
        )
        .values(is_cancelled=True)
    )
    await db.commit()
    return {"success": True}


@app.delete("/api/reminders/{reminder_id}")
async def delete_reminder(
    reminder_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel (soft-delete) a reminder so it no longer appears in the list."""
    reminder = await db.get(Reminder, reminder_id)
    if not reminder or reminder.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Reminder not found")
    reminder.is_cancelled = True
    await db.commit()
    return {"success": True}


@app.get("/api/recurrences")
async def list_recurrences(
    group_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return active recurring reminders scoped to personal brain or a specific group."""
    stmt = (
        select(Recurrence)
        .outerjoin(Message, Recurrence.message_id == Message.id)
        .where(Recurrence.user_id == current_user.id, Recurrence.is_active == True)
    )
    if group_id is not None:
        stmt = stmt.where(Message.group_id == group_id)
    else:
        stmt = stmt.where(or_(Recurrence.message_id == None, Message.group_id == None))
    stmt = stmt.order_by(Recurrence.created_at.desc())
    result = await db.execute(stmt)
    recs = result.scalars().all()
    items = []
    for r in recs:
        items.append({
            "id":          r.id,
            "task":        r.template_content,
            "rule":        r.rule,
            "time_of_day": r.time_of_day,
            "day_of_week": r.day_of_week,
            "next_fire":   r.next_fire.isoformat() if r.next_fire else None,
            "last_fired":  r.last_fired.isoformat() if r.last_fired else None,
            "is_active":   r.is_active,
        })
    return {"success": True, "recurrences": items}


@app.delete("/api/recurrences/{recurrence_id}")
async def delete_recurrence(
    recurrence_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Cancel a recurring reminder so it no longer fires."""
    rec = await db.get(Recurrence, recurrence_id)
    if not rec or rec.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Recurring reminder not found")
    rec.is_active = False
    await db.commit()
    return {"success": True}


@app.patch("/api/messages/{message_id}/remind-at")
async def set_remind_at(
    message_id: int,
    body: RemindAtRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Set (or update) a reminder time for a saved To-Do that has no time yet."""
    msg = await db.scalar(
        select(Message).where(Message.id == message_id, Message.user_id == current_user.id)
    )
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")

    # Parse the ISO datetime the iOS app sends (e.g. "2026-06-02T18:00:00+05:30")
    try:
        remind_dt = datetime.fromisoformat(body.remind_at)
    except ValueError:
        raise HTTPException(status_code=422, detail="remind_at must be an ISO 8601 datetime")

    # Decompose into date + time for reminder_service
    from zoneinfo import ZoneInfo as _ZI
    user_tz = current_user.timezone or "Asia/Kolkata"
    local_dt = remind_dt.astimezone(_ZI(user_tz))
    analysis = {
        "event_time":  local_dt.strftime("%H:%M"),
        "due_date":    local_dt.strftime("%Y-%m-%d"),
        "priority":    "normal",
        "actionables": [msg.content[:120]],
        "essence":     msg.summary or msg.content[:80],
    }

    reminder = await reminder_service.create(
        user=current_user,
        content=msg.content,
        analysis=analysis,
        message_id=message_id,
        db=db,
    )
    if not reminder:
        raise HTTPException(status_code=500, detail="Could not create reminder")

    import json as _json_ra
    new_tags = {"remind_at": reminder.remind_at.isoformat(), "needs_time": False}
    await db.execute(
        text("UPDATE messages SET tags = tags || CAST(:extra AS jsonb) WHERE id = :mid")
        .bindparams(extra=_json_ra.dumps(new_tags), mid=message_id)
    )
    await db.commit()

    return {
        "success":     True,
        "remind_at":   reminder.remind_at.isoformat(),
        "reminder_id": reminder.id,
    }


@app.post("/api/messages/capture")
async def capture_message(
    message: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    group_id = message.group_id
    content  = message.content

    # ── Group context: parse ALL @mentions before AI processing ─────
    assignments: list[dict] = []
    members: list[dict] = []
    if group_id:
        members = await grp_svc.get_group_members(group_id, db)
        assignments = grp_svc.parse_all_mentions(content, members)

    result = await message_processor.process(
        user_phone=current_user.phone_number, content=content,
        message_type=message.message_type, media_url=message.media_url, db=db,
        skip_query=True,
        group_id=group_id,
        # @mention tasks are always To-Do — skip classifier/LLM entirely
        force_bucket="Track" if message.expense_amount is not None else ("To-Do" if assignments else (message.force_bucket or None)),
        # All group captures: use classifier or rule-based fallback, never LLM
        no_llm_fallback=bool(group_id),
        # When the task is assigned to someone, skip sender's reminder — assignee gets it instead
        skip_reminder=bool(assignments),
    )

    # ── Tag message with group_id / assignments ───────────────────────
    if group_id and result.get("message_id"):
        # Single-assignee field kept for backwards compatibility
        primary_uid = assignments[0]["user_id"] if assignments else None

        # Detect group-wide reminder: no specific @mention but a due time was extracted
        is_group_reminder = (not assignments) and bool(result.get("due_date") or result.get("remind_at"))

        import json as _json

        extra_tags: dict = {}
        if assignments:
            extra_tags["assignments"] = assignments
        if is_group_reminder:
            extra_tags["group_reminder"] = True

        # Single UPDATE: set group_id, assigned_to_user_id, and merge extra tags atomically
        await db.execute(
            text(
                "UPDATE messages SET group_id = :gid, assigned_to_user_id = :auid,"
                " tags = tags || CAST(:extra AS jsonb) WHERE id = :mid"
            ).bindparams(
                gid=group_id,
                auid=primary_uid,
                extra=_json.dumps(extra_tags),
                mid=result["message_id"],
            )
        )
        await db.commit()

        result["group_id"]       = group_id
        result["assigned_to"]    = assignments[0]["name"] if assignments else None
        result["assigned_to_id"] = primary_uid
        result["assignments"]    = assignments

        # ── Bust bootstrap cache for sender + each assignee ──────────
        from services import redis_cache as _rc
        asyncio.create_task(_rc.cache_del(_rc.bootstrap_key(current_user.id, group_id)))
        for _a in assignments:
            asyncio.create_task(_rc.cache_del(_rc.bootstrap_key(_a["user_id"], None)))

        # ── Broadcast new message to group WebSocket ──────────────────
        asyncio.create_task(ws_manager.broadcast(group_id, {
            "type":                "new_message",
            "id":                  result.get("message_id"),
            "content":             content,
            "essence":             result.get("essence", content[:80]),
            "message_type":        "text",
            "media_url":           message.media_url,
            "category":            result.get("category", "Random"),
            "all_buckets":         [result.get("category", "Random")],
            "tags":                extra_tags,
            "created_at":          datetime.utcnow().isoformat(),
            "due_date":            result.get("due_date"),
            "group_id":            group_id,
            "sender_id":           current_user.id,
            "sender_name":         current_user.name,
            "assigned_to_user_id": primary_uid,
            "assignments":         assignments,
            "starred":             False,
        }, exclude_user_id=current_user.id))

        # ── Mirror To-Do in each assignee's personal feed ────────────
        mirror_objects: list[tuple[int, Message]] = []  # (auid, mirror)
        for assignment in assignments:
            auid = assignment["user_id"]
            if auid == current_user.id:
                continue
            todo_tags = {
                "primary_bucket":    "To-Do",
                "intent_bucket":     "To-Do",
                "due_date":          result.get("due_date"),
                "assigned_by":       current_user.name,
                "assigned_by_id":    current_user.id,
                "group_id":          group_id,
                "source_message_id": result["message_id"],
            }
            mirror = Message(
                user_id=auid,
                content=content,
                message_type=MessageType.TEXT,
                summary=result.get("essence") or content[:80],
                tags=todo_tags,
                assigned_to_user_id=auid,
            )
            db.add(mirror)
            mirror_objects.append((auid, mirror))

        # Flush all mirrors in one shot so mirror IDs are available
        await db.flush()

        # ── Create reminders for assignees (not the sender) ──────────
        _evt_time = result.get("event_time")
        _due_date = result.get("due_date")
        if _evt_time:
            for auid, mirror in mirror_objects:
                try:
                    assignee_user = await db.get(User, auid)
                    if assignee_user:
                        await reminder_service.create(
                            user=assignee_user,
                            content=content,
                            analysis={
                                "event_time":  _evt_time,
                                "due_date":    _due_date,
                                "priority":    result.get("priority", "normal"),
                                "actionables": [content],
                                "essence":     content,
                            },
                            message_id=mirror.id,
                            db=db,
                        )
                except Exception as e:
                    print(f"[reminder] assignee reminder failed for uid={auid}: {e}")

        _push_assign: list[tuple[int, list[str]]] = []
        for assignment in assignments:
            auid = assignment["user_id"]
            if auid == current_user.id:
                continue
            try:
                rows = await db.execute(
                    select(DeviceToken.token).where(DeviceToken.user_id == auid)
                )
                _push_assign.append((auid, [r[0] for r in rows.all()]))
            except Exception as e:
                print(f"[push] token fetch failed for uid={auid}: {e}")

        _push_grp: list[tuple[int, list[str]]] = []
        if is_group_reminder:
            for member in members:
                muid = member["id"]
                if muid == current_user.id:
                    continue
                try:
                    rows = await db.execute(
                        select(DeviceToken.token).where(DeviceToken.user_id == muid)
                    )
                    _push_grp.append((muid, [r[0] for r in rows.all()]))
                except Exception as e:
                    print(f"[push] group reminder token fetch failed for uid={muid}: {e}")

        await db.commit()

        # ── Fire all notifications in the background ──────────────────
        # Captured by value so the HTTP response is returned immediately.
        _sender    = current_user.name
        _gid       = group_id
        _mid       = result["message_id"]
        _body      = content[:80]

        async def _send_pushes() -> None:
            from services.group_service import total_unread_for_user as _tuu
            for auid, tokens in _push_assign:
                try:
                    async with async_session_maker() as _s:
                        badge = await _tuu(_s, auid)
                    for token in tokens:
                        await send_apns_notification(
                            device_token=token,
                            title=f"{_sender} assigned you a task",
                            body=_body, badge=badge,
                            data={"type": "assignment", "group_id": _gid,
                                  "message_id": _mid},
                        )
                except Exception as e:
                    print(f"[push] assignment notify failed for uid={auid}: {e}")
            for muid, tokens in _push_grp:
                try:
                    async with async_session_maker() as _s:
                        badge = await _tuu(_s, muid)
                    for token in tokens:
                        await send_apns_notification(
                            device_token=token,
                            title=f"Reminder · {_sender}",
                            body=_body, badge=badge,
                            data={"type": "group_reminder", "group_id": _gid,
                                  "message_id": _mid},
                        )
                except Exception as e:
                    print(f"[push] group reminder notify failed for uid={muid}: {e}")

        asyncio.create_task(_send_pushes())

    # ── Temporal parsing: LLM extracts time/date for all reminder + recurring captures ──
    # Runs for any personal capture that has reminder keywords or recurring signals.
    # One LLM call handles both one-time reminder time correction AND recurrence creation.
    _REMINDER_KW = {"remind", "reminder", "don't forget", "alert", "notify", "ping"}
    _lc = content.lower()
    _has_temporal = (
        not group_id
        and (
            recurrence_service.is_recurring(content)
            or result.get("remind_at")
            or any(kw in _lc for kw in _REMINDER_KW)
        )
        and result.get("message_id")
    )
    if _has_temporal:
        try:
            from datetime import datetime as _dt
            from zoneinfo import ZoneInfo as _ZI
            _now_ist = _dt.now(_ZI("Asia/Kolkata"))
            _today   = _now_ist.strftime("%Y-%m-%d")
            _now_str = _now_ist.strftime("%H:%M")

            temporal = await recurrence_service.parse_temporal(content, _today, _now_str)

            if temporal:
                # 1. Correct one-time reminder time with LLM result
                if result.get("reminder_id") and temporal.get("remind_at_time"):
                    r_date = temporal.get("remind_at_date") or _today
                    r_h, r_m = map(int, temporal["remind_at_time"].split(":"))
                    _tz = _ZI(current_user.timezone or "Asia/Kolkata")
                    _local = _dt.fromisoformat(r_date).replace(
                        hour=r_h, minute=r_m, second=0, microsecond=0, tzinfo=_tz
                    )
                    if _local <= _now_ist.replace(tzinfo=_tz):
                        _local += timedelta(days=1)
                    _utc = _local.astimezone(_ZI("UTC")).replace(tzinfo=None)
                    await db.execute(
                        update(Reminder)
                        .where(Reminder.id == result["reminder_id"])
                        .values(remind_at=_utc)
                    )
                    await db.commit()
                    result["remind_at"] = _utc.isoformat()
                    print(f"[temporal] corrected reminder #{result['reminder_id']} → {_utc} UTC")

                # 2. Create recurrence(s) if recurring
                #    Defense in depth: only honour the LLM's recurrence verdict when
                #    the text actually contains an explicit repetition signal. Stops
                #    a hallucinated "daily" rule from a one-time "...today" capture.
                if (
                    temporal.get("is_recurring")
                    and temporal.get("recurrence_rule")
                    and recurrence_service.is_recurring(content)
                ):
                    _rule      = temporal["recurrence_rule"]
                    _time      = temporal.get("time_of_day") or "09:00"
                    _task      = temporal.get("task") or content
                    _mid       = result["message_id"]
                    _days      = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                    _recs      = []

                    if _rule == "multi-weekly" and temporal.get("recurrence_days"):
                        for _dow in temporal["recurrence_days"]:
                            _rec = await recurrence_service.create(
                                current_user.id,
                                {"rule": "weekly", "day_of_week": _dow,
                                 "time_of_day": _time, "task": _task},
                                content, db,
                            )
                            if _rec:
                                _recs.append(_rec)
                                await db.execute(
                                    update(Recurrence).where(Recurrence.id == _rec.id)
                                    .values(message_id=_mid)
                                )
                        _dow_names = ", ".join(
                            _days[d] for d in sorted(temporal["recurrence_days"])
                        )
                        _rule_str = f"Repeats {_dow_names} at {_time} IST"
                    else:
                        _rec = await recurrence_service.create(
                            current_user.id,
                            {"rule": _rule,
                             "day_of_week": temporal.get("day_of_week"),
                             "time_of_day": _time, "task": _task},
                            content, db,
                        )
                        if _rec:
                            _recs.append(_rec)
                            await db.execute(
                                update(Recurrence).where(Recurrence.id == _rec.id)
                                .values(message_id=_mid)
                            )
                        _rule_str = recurrence_service._rule_display(_recs[0]) if _recs else ""

                    if _recs:
                        result["recurring"]       = True
                        result["recurrence_rule"] = _rule_str
                        result["recurrence_id"]   = _recs[0].id
                        if result.get("essence"):
                            result["essence"] = f"{result['essence']} · {_rule_str}"
                        # Cancel the companion one-time reminder
                        await db.execute(
                            update(Reminder).where(Reminder.message_id == _mid)
                            .values(is_cancelled=True)
                        )
                        await db.commit()
                        print(f"[temporal] created {len(_recs)} recurrence(s): {_rule_str}")

        except Exception as e:
            print(f"[capture] Temporal parsing failed: {e}")

    # ── Needs-time flag: reminder keyword but no time found by LLM or regex ──
    if (
        result.get("category") == "To-Do"
        and not result.get("remind_at")
        and any(kw in _lc for kw in _REMINDER_KW)
        and result.get("message_id")
        and not group_id
    ):
        result["needs_time"] = True
        import json as _json_nt
        await db.execute(
            text("UPDATE messages SET tags = tags || '{\"needs_time\": true}'::jsonb WHERE id = :mid")
            .bindparams(mid=result["message_id"])
        )
        await db.commit()

    # ── Expense metadata ─────────────────────────────────────────────
    if message.expense_amount is not None and result.get("message_id"):
        import json as _json_exp
        # Always record who paid — default to the person capturing the expense
        payer_id   = message.expense_payer_id   if message.expense_payer_id   is not None else current_user.id
        payer_name = message.expense_payer_name if message.expense_payer_name else current_user.name
        expense_tags = {
            "expense_amount":   message.expense_amount,
            "expense_category": message.expense_category or "Others",
            "expense_payer_id":   payer_id,
            "expense_payer_name": payer_name,
        }
        if message.expense_context:
            expense_tags["expense_context"] = message.expense_context
        await db.execute(
            text("UPDATE messages SET tags = tags || CAST(:extra AS jsonb) WHERE id = :mid")
            .bindparams(extra=_json_exp.dumps(expense_tags), mid=result["message_id"])
        )
        await db.commit()
        result["expense_amount"]   = message.expense_amount
        result["expense_category"] = message.expense_category or "Others"
        result["expense_payer_id"]   = payer_id
        result["expense_payer_name"] = payer_name
        if message.expense_context:
            result["expense_context"] = message.expense_context

    return {"success": True, "message": "Content captured successfully", "data": result}


@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload an image; stores binary data in the DB (stored_images table).
    Returns a URL like /api/images/{id} that the iOS app can load via AsyncImage.
    No external storage service required.
    """
    data         = await file.read()
    mime_type    = file.content_type or "image/jpeg"

    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    if len(data) > 10 * 1024 * 1024:   # 10 MB hard cap
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    img = StoredImage(user_id=current_user.id, data=data, mime_type=mime_type)
    db.add(img)
    await db.commit()
    await db.refresh(img)

    # Return an absolute URL so the iOS app can load it via AsyncImage
    base_url = os.getenv("RAILWAY_PUBLIC_DOMAIN", "")
    if base_url:
        url = f"https://{base_url}/api/images/{img.id}"
    else:
        url = f"/api/images/{img.id}"

    return {"url": url}


@app.get("/api/images/{image_id}")
async def serve_image(
    image_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Serve a stored image by ID. Auth-gated to the owning user."""
    from fastapi.responses import Response
    img = await db.get(StoredImage, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    if img.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your image")
    return Response(content=img.data, media_type=img.mime_type)


@app.post("/api/search")
async def search_messages(
    search: SearchQuery,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if search.group_id:
        # Verify membership before searching the group
        group = await grp_svc.get_group_by_id(search.group_id, current_user.id, db)
        if not group:
            raise HTTPException(status_code=403, detail="Not a member of this group")
    search_data = await search_service.search(
        user_phone=current_user.phone_number, query=search.query,
        limit=search.limit, category_filter=search.category_filter,
        group_id=search.group_id, db=db, fast=search.fast,
    )
    return {
        "success":          True,
        "query":            search.query,
        "natural_response": search_data.get("natural_response", ""),
        "results":          search_data.get("results", []),
        "total":            len(search_data.get("results", [])),
    }


@app.patch("/api/messages/{message_id}/done")
async def mark_message_done(
    message_id: int,
    body: Optional[DoneRequest] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import update as sa_update
    done_val = body.done if body is not None else True

    # Fetch the message first so we can:
    # 1. Allow group members to mark group-wide tasks done (not just the owner)
    # 2. Bust bootstrap caches for affected users after the update
    msg_row = await db.execute(
        select(Message).where(Message.id == message_id)
    )
    msg = msg_row.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")

    # Authorization: owner always allowed; group member allowed for unassigned group-wide tasks
    is_owner = msg.user_id == current_user.id
    is_group_member_task = False
    if not is_owner and msg.group_id and not msg.assigned_to_user_id:
        member_check = await db.execute(
            select(GroupMember.id).where(
                and_(GroupMember.group_id == msg.group_id, GroupMember.user_id == current_user.id)
            )
        )
        is_group_member_task = member_check.scalar_one_or_none() is not None

    if not is_owner and not is_group_member_task:
        raise HTTPException(status_code=403, detail="Not allowed")

    jsonb_expr = f"jsonb_set(COALESCE(tags, '{{}}'), '{{done}}', '{str(done_val).lower()}'::jsonb)"
    await db.execute(
        sa_update(Message)
        .where(Message.id == message_id)
        .values(tags=text(jsonb_expr))
    )
    await db.commit()

    from services import redis_cache as _rc
    # Always bust the owner's bootstrap cache
    asyncio.create_task(_rc.cache_del(_rc.bootstrap_key(msg.user_id, None)))
    if msg.group_id:
        asyncio.create_task(_rc.cache_del(_rc.bootstrap_key(msg.user_id, msg.group_id)))

    # For group-wide tasks, bust all members' personal bootstrap caches and broadcast
    # so their "Assigned to Me" updates without needing a manual pull-to-refresh.
    is_group_wide_task = msg.group_id and not msg.assigned_to_user_id
    if (is_group_member_task) or (is_owner and is_group_wide_task):
        try:
            all_members = await db.execute(
                select(GroupMember.user_id).where(GroupMember.group_id == msg.group_id)
            )
            for (muid,) in all_members.all():
                asyncio.create_task(_rc.cache_del(_rc.bootstrap_key(muid, None)))
        except Exception:
            pass
        asyncio.create_task(ws_manager.broadcast(msg.group_id, {
            "type":       "todo_completed",
            "message_id": message_id,
            "done":       done_val,
            "done_by":    current_user.name,
        }))

    return {"success": True}


@app.patch("/api/messages/{message_id}/bucket")
async def update_message_bucket(
    message_id: int,
    body: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import update as sa_update
    VALID_BUCKETS = {"Remember", "To-Do", "Ideas", "Track", "Events", "Random"}
    bucket = body.get("bucket", "")
    if bucket not in VALID_BUCKETS:
        raise HTTPException(status_code=400, detail=f"Invalid bucket. Must be one of: {', '.join(sorted(VALID_BUCKETS))}")

    # Fetch message content before updating (needed for annotation)
    msg_result = await db.execute(
        select(Message).where(and_(Message.id == message_id, Message.user_id == current_user.id))
    )
    msg = msg_result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=404, detail="Message not found")

    # Python-side tag manipulation: update bucket and set due_date when
    # promoting to To-Do so the task surfaces under TODAY/SOMEDAY rather
    # than being invisible (tags.due_date is None for Remember/Ideas/etc.)
    updated_tags = dict(msg.tags or {})
    updated_tags["primary_bucket"] = bucket
    updated_tags["all_buckets"]    = [bucket]
    if bucket == "To-Do" and not updated_tags.get("due_date"):
        updated_tags["due_date"] = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")

    await db.execute(
        sa_update(Message)
        .where(and_(Message.id == message_id, Message.user_id == current_user.id))
        .values(tags=updated_tags)
    )

    # Store as training annotation — user corrections are ground truth
    annotation = LabelAnnotation(
        user_id=current_user.id,
        message_id=message_id,
        text=msg.content,
        label=bucket,
        source="user_correction",
    )
    db.add(annotation)
    await db.commit()

    # Bust bootstrap cache so the next app-load reflects the new bucket
    from services import redis_cache as _rc
    asyncio.create_task(_rc.cache_del(_rc.bootstrap_key(current_user.id, None)))
    if msg.group_id:
        asyncio.create_task(_rc.cache_del(_rc.bootstrap_key(current_user.id, msg.group_id)))

    return {"success": True}


@app.patch("/api/messages/{message_id}/content")
async def update_message_content(
    message_id: int,
    body: dict,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import update as sa_update
    new_content = (body.get("content") or "").strip()
    if not new_content:
        raise HTTPException(status_code=422, detail="content is required")

    result = await db.execute(
        sa_update(Message)
        .where(and_(Message.id == message_id, Message.user_id == current_user.id))
        .values(content=new_content, summary=new_content)
    )
    await db.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Message not found")

    from services import redis_cache as _rc
    asyncio.create_task(_rc.cache_del(_rc.bootstrap_key(current_user.id, None)))
    asyncio.create_task(_rc.cache_del_user_searches(current_user.id))

    return {"success": True}


@app.get("/api/annotations/export")
async def export_annotations(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = 5000,
):
    """
    Export label annotations for retraining the ONNX intent classifier.
    Returns all user-correction annotations (deduplicated, most recent label wins).
    Used by retrain.py in the intent-classifier-poc repo.
    """
    result = await db.execute(
        select(LabelAnnotation)
        .order_by(LabelAnnotation.created_at.desc())
        .limit(limit)
    )
    rows = result.scalars().all()

    # Deduplicate by text — most recent annotation wins (already ordered desc)
    seen:    dict[str, dict] = {}
    for row in rows:
        key = row.text.strip().lower()
        if key not in seen:
            seen[key] = {"text": row.text, "label": row.label, "source": row.source}

    return list(seen.values())


@app.patch("/api/messages/{message_id}/items/{item_index}/complete")
async def complete_list_item(
    message_id: int,
    item_index: int,
    current_user: User = Depends(get_current_user),
):
    from services.list_service import ListService
    ls = ListService(cerebras_client)
    success = await ls.complete_item(message_id, item_index)
    if not success:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"success": True}


@app.post("/api/messages/{message_id}/items")
async def add_list_item(
    message_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
):
    body = await request.json()
    task = (body.get("task") or "").strip()
    if not task:
        raise HTTPException(status_code=422, detail="task is required")
    from services.list_service import ListService
    ls    = ListService(cerebras_client)
    index = await ls.add_item(message_id, task)
    if index is None:
        raise HTTPException(status_code=404, detail="List not found")
    return {"success": True, "item_index": index}


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


@app.get("/api/admin/stats")
async def admin_stats(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Admin-only endpoint — protected by ADMIN_SECRET env var."""
    admin_secret = os.getenv("ADMIN_SECRET", "")
    provided     = (
        request.headers.get("X-Admin-Secret")
        or request.query_params.get("admin_secret")
        or ""
    )
    if not admin_secret or provided != admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden")

    now = datetime.utcnow()
    day_ago   = now - timedelta(days=1)
    week_ago  = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    # ── Totals ──────────────────────────────────────────────────
    total_users   = await db.scalar(select(func.count(User.id)))
    total_msgs    = await db.scalar(select(func.count(Message.id)))
    active_today  = await db.scalar(
        select(func.count(func.distinct(Message.user_id)))
        .where(Message.created_at >= day_ago)
    )
    active_week   = await db.scalar(
        select(func.count(func.distinct(Message.user_id)))
        .where(Message.created_at >= week_ago)
    )
    new_users_week = await db.scalar(
        select(func.count(User.id)).where(User.created_at >= week_ago)
    )

    # ── Per-user stats ──────────────────────────────────────────
    user_rows = await db.execute(
        select(
            User.id, User.name, User.phone_number, User.email,
            User.created_at, User.last_login,
            User.telegram_chat_id,
            func.count(Message.id).label("msg_count"),
            func.max(Message.created_at).label("last_active"),
        )
        .outerjoin(Message, Message.user_id == User.id)
        .group_by(User.id)
        .order_by(func.count(Message.id).desc())
    )
    users = []
    for row in user_rows.all():
        users.append({
            "id":           row.id,
            "name":         row.name,
            "phone":        row.phone_number,
            "email":        row.email,
            "joined":       row.created_at.isoformat() if row.created_at else None,
            "last_login":   row.last_login.isoformat() if row.last_login else None,
            "last_active":  row.last_active.isoformat() if row.last_active else None,
            "msg_count":    row.msg_count,
            "has_telegram": bool(row.telegram_chat_id),
        })

    # ── Feature (bucket) usage ──────────────────────────────────
    # primary_bucket is stored in tags JSONB
    bucket_rows = await db.execute(
        text("""
            SELECT
                COALESCE(
                    tags->>'primary_bucket',
                    tags->>'intent_bucket',
                    'Unknown'
                ) AS bucket,
                COUNT(*) AS cnt
            FROM messages
            GROUP BY bucket
            ORDER BY cnt DESC
        """)
    )
    feature_usage = {row.bucket: row.cnt for row in bucket_rows.all()}

    # ── Media type breakdown ─────────────────────────────────────
    type_rows = await db.execute(
        select(Message.message_type, func.count(Message.id))
        .group_by(Message.message_type)
    )
    media_types = {
        (row[0].value if row[0] else "unknown"): row[1]
        for row in type_rows.all()
    }

    # ── Messages per day — last 14 days ─────────────────────────
    daily_rows = await db.execute(
        text("""
            SELECT
                DATE(created_at) AS day,
                COUNT(*)         AS cnt
            FROM messages
            WHERE created_at >= NOW() - INTERVAL '14 days'
            GROUP BY day
            ORDER BY day
        """)
    )
    daily_activity = {str(row.day): row.cnt for row in daily_rows.all()}

    # ── Messages per day per user — last 7 days ──────────────────
    user_daily_rows = await db.execute(
        text("""
            SELECT
                m.user_id,
                DATE(m.created_at) AS day,
                COUNT(*)           AS cnt
            FROM messages m
            WHERE m.created_at >= NOW() - INTERVAL '7 days'
            GROUP BY m.user_id, day
            ORDER BY day, m.user_id
        """)
    )
    user_daily: dict = {}
    for row in user_daily_rows.all():
        uid = str(row.user_id)
        if uid not in user_daily:
            user_daily[uid] = {}
        user_daily[uid][str(row.day)] = row.cnt

    # ── Group analytics ──────────────────────────────────────────
    total_pro_accounts = await db.scalar(select(func.count(ProAccount.id))) or 0
    total_groups       = await db.scalar(select(func.count(Group.id))) or 0
    total_group_msgs   = await db.scalar(
        select(func.count(Message.id)).where(Message.group_id.isnot(None))
    ) or 0
    total_assigned     = await db.scalar(
        select(func.count(Message.id)).where(Message.assigned_to_user_id.isnot(None))
    ) or 0

    # Per-group stats
    group_rows = await db.execute(
        text("""
            SELECT
                g.id,
                g.name,
                g.emoji,
                g.created_at,
                u.name            AS owner_name,
                COUNT(DISTINCT gm.user_id)   AS member_count,
                COUNT(DISTINCT m.id)         AS msg_count,
                MAX(m.created_at)            AS last_active
            FROM groups g
            JOIN pro_accounts pa ON pa.id = g.account_id
            JOIN users u         ON u.id  = pa.owner_id
            LEFT JOIN group_members gm ON gm.group_id = g.id
            LEFT JOIN messages m       ON m.group_id  = g.id
            GROUP BY g.id, g.name, g.emoji, g.created_at, u.name
            ORDER BY msg_count DESC
        """)
    )
    groups_detail = []
    for row in group_rows.all():
        groups_detail.append({
            "id":           row.id,
            "name":         row.name,
            "emoji":        row.emoji or "👥",
            "owner":        row.owner_name,
            "member_count": row.member_count,
            "msg_count":    row.msg_count,
            "last_active":  row.last_active.isoformat() if row.last_active else None,
            "created_at":   row.created_at.isoformat() if row.created_at else None,
        })

    # Per-pro-account: owner + how many members invited
    pro_rows = await db.execute(
        text("""
            SELECT
                u.name              AS owner_name,
                u.phone_number      AS owner_phone,
                pa.plan_type,
                pa.created_at,
                COUNT(DISTINCT pam.id)     AS invited_count,
                SUM(CASE WHEN pam.status = 'active' THEN 1 ELSE 0 END) AS active_members,
                COUNT(DISTINCT g.id)       AS group_count
            FROM pro_accounts pa
            JOIN users u ON u.id = pa.owner_id
            LEFT JOIN pro_account_members pam ON pam.account_id = pa.id
            LEFT JOIN groups g ON g.account_id = pa.id
            GROUP BY pa.id, u.name, u.phone_number, pa.plan_type, pa.created_at
            ORDER BY pa.created_at DESC
        """)
    )
    pro_accounts_detail = []
    for row in pro_rows.all():
        pro_accounts_detail.append({
            "owner":          row.owner_name,
            "phone":          row.owner_phone,
            "plan_type":      row.plan_type,
            "invited":        row.invited_count,
            "active_members": row.active_members or 0,
            "group_count":    row.group_count,
            "created_at":     row.created_at.isoformat() if row.created_at else None,
        })

    # Group message activity — last 14 days
    group_daily_rows = await db.execute(
        text("""
            SELECT DATE(created_at) AS day, COUNT(*) AS cnt
            FROM messages
            WHERE group_id IS NOT NULL
              AND created_at >= NOW() - INTERVAL '14 days'
            GROUP BY day ORDER BY day
        """)
    )
    group_daily_activity = {str(row.day): row.cnt for row in group_daily_rows.all()}

    saves_count = total_msgs or 0

    return {
        "summary": {
            "total_users":    total_users,
            "total_messages": total_msgs,
            "active_today":   active_today,
            "active_week":    active_week,
            "new_users_week": new_users_week,
            "avg_msgs_per_user": round(total_msgs / total_users, 1) if total_users else 0,
        },
        "users":          users,
        "feature_usage":  feature_usage,
        "media_types":    media_types,
        "daily_activity": daily_activity,
        "user_daily":     user_daily,
        "groups": {
            "total_pro_accounts":  total_pro_accounts,
            "total_groups":        total_groups,
            "total_group_messages": total_group_msgs,
            "total_assigned_tasks": total_assigned,
            "pro_accounts":        pro_accounts_detail,
            "groups_detail":       groups_detail,
            "group_daily_activity": group_daily_activity,
        },
    }


@app.get("/api/analytics")
async def get_user_analytics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    phone = current_user.phone_number
    total = await db.scalar(
        select(func.count(Message.id)).join(User).where(User.phone_number == phone)
    )
    cat_stats = await db.execute(
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
            "platform":         "telegram",
            "webhook_endpoint": "/webhook/telegram",
            "current_webhook":  info.get("result", {}),
            "configured_url":   Config.TELEGRAM_WEBHOOK_URL,
        }
    except Exception:
        return {
            "platform":       "telegram",
            "configured_url": Config.TELEGRAM_WEBHOOK_URL,
            "status":         "not_configured",
        }


# ================== Pro Plan Endpoints ==================

class InviteRequest(BaseModel):
    phone_number: Optional[str] = None
    email: Optional[str] = None

class AcceptInviteRequest(BaseModel):
    token: str

class CreateGroupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None
    emoji: Optional[str] = None
    photo_url: Optional[str] = None

class UpdateGroupPhotoRequest(BaseModel):
    photo_url: Optional[str] = None

class UpdateGroupNameRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)

class AddGroupMemberRequest(BaseModel):
    user_id: int

class AddGroupMemberByPhoneRequest(BaseModel):
    phone_number: str


@app.get("/api/pro/status")
async def get_pro_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    acct = await grp_svc.get_pro_account_for_user(current_user.id, db)
    if not acct:
        return {
            "is_pro": current_user.is_pro,
            "account": None,
            "members": [],
        }
    members = await grp_svc.get_account_members(acct.id, db)
    return {
        "is_pro": current_user.is_pro,
        "account": {
            "id": acct.id,
            "plan_type": acct.plan_type,
            "max_members": acct.max_members,
            "expires_at": acct.expires_at.isoformat() if acct.expires_at else None,
        },
        "members": members,
    }


@app.post("/api/pro/invite")
async def invite_member(
    req: InviteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    identifier = req.email or req.phone_number
    if not identifier:
        raise HTTPException(status_code=400, detail="Provide a phone number or email address.")
    result = await grp_svc.invite_member(current_user, identifier, db)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.get("/api/pro/my-invites")
async def my_pending_invites(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rows = (await db.execute(
        select(ProAccountMember).where(
            ProAccountMember.phone_number == current_user.phone_number,
            ProAccountMember.status == "pending",
        )
    )).scalars().all()

    result = []
    for row in rows:
        inviter = await db.get(User, row.invited_by)
        result.append({
            "token": row.invite_token,
            "invited_by_name": inviter.name if inviter else "Someone",
            "invited_at": row.invited_at.isoformat() if row.invited_at else None,
        })

    return {"success": True, "invites": result}


@app.post("/api/pro/accept-invite")
async def accept_invite(
    req: AcceptInviteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await grp_svc.accept_invite(req.token, current_user, db)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


# ================== Groups Endpoints ==================

@app.get("/api/groups")
async def list_groups(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    groups = await grp_svc.get_user_groups(current_user.id, db)
    return {"success": True, "groups": groups}


@app.post("/api/groups")
async def create_group(
    req: CreateGroupRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await grp_svc.create_group(current_user, req.name, req.description, db)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    # Optionally store emoji and/or photo_url
    if (req.emoji or req.photo_url) and result.get("group_id"):
        grp = await db.get(Group, result["group_id"])
        if grp:
            if req.emoji:
                grp.emoji = req.emoji
            if req.photo_url:
                grp.photo_url = req.photo_url
            await db.commit()
    return result


@app.patch("/api/groups/{group_id}/photo")
async def update_group_photo(
    group_id: int,
    req: UpdateGroupPhotoRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the group photo (or clear it by passing null). Admin only."""
    group = await grp_svc.get_group_by_id(group_id, current_user.id, db)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found or access denied")
    if not await grp_svc.is_group_admin(group_id, current_user.id, db):
        raise HTTPException(status_code=403, detail="Only the group owner can change the group photo.")
    group.photo_url = req.photo_url
    await db.commit()
    return {"success": True, "photo_url": group.photo_url}


@app.patch("/api/groups/{group_id}/name")
async def update_group_name(
    group_id: int,
    req: UpdateGroupNameRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update the group name. Admin only."""
    group = await grp_svc.get_group_by_id(group_id, current_user.id, db)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found or access denied")
    if not await grp_svc.is_group_admin(group_id, current_user.id, db):
        raise HTTPException(status_code=403, detail="Only the group owner can change the group name.")
    group.name = req.name
    await db.commit()
    return {"success": True, "name": group.name}


@app.get("/api/groups/{group_id}/messages")
async def get_group_messages(
    group_id: int,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    group = await grp_svc.get_group_by_id(group_id, current_user.id, db)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found or access denied")
    messages = await grp_svc.get_group_messages(group_id, limit, db)
    return {"success": True, "group": {"id": group.id, "name": group.name, "emoji": group.emoji}, "messages": messages}


@app.get("/api/groups/{group_id}/members")
async def get_group_members(
    group_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    group = await grp_svc.get_group_by_id(group_id, current_user.id, db)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found or access denied")
    members = await grp_svc.get_group_members(group_id, db)
    return {"success": True, "members": members}


@app.post("/api/groups/{group_id}/members")
async def add_group_member(
    group_id: int,
    req: AddGroupMemberRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Add an existing user to a group by user_id. Only the group admin (its Pro
    owner) may add members; the invitee needs no Pro plan. Subject to the cap."""
    group = await grp_svc.get_group_by_id(group_id, current_user.id, db)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found or access denied")
    if not await grp_svc.is_group_admin(group_id, current_user.id, db):
        raise HTTPException(status_code=403, detail="Only the group owner can add members.")
    target = await db.get(User, req.user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        added = await grp_svc.add_member_to_group(group, req.user_id, "member", db)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"success": True, "added": added, "name": target.name}


@app.post("/api/groups/{group_id}/invite")
async def invite_group_member_by_phone(
    group_id: int,
    req: AddGroupMemberByPhoneRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Add a group member by phone number. Only the group admin (its Pro owner) may
    add members; the invitee needs no Pro plan. If no account exists for the number
    yet, returns user_exists=false so the client can prompt sharing the invite link."""
    group = await grp_svc.get_group_by_id(group_id, current_user.id, db)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found or access denied")
    if not await grp_svc.is_group_admin(group_id, current_user.id, db):
        raise HTTPException(status_code=403, detail="Only the group owner can add members.")
    result = await grp_svc.add_member_by_phone(group, req.phone_number, db)
    if not result["success"]:
        # user_exists=False is a normal "share the link" case, not a hard error,
        # but full groups should surface as a 409.
        if result.get("user_exists") and "full" in result["message"].lower():
            raise HTTPException(status_code=409, detail=result["message"])
        return result
    return result


@app.get("/api/groups/{group_id}/invite-link")
async def get_group_invite_link(
    group_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the shareable invite link. Only the group admin (its Pro owner) may
    fetch/share it — that is how members get added."""
    group = await grp_svc.get_group_by_id(group_id, current_user.id, db)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found or access denied")
    if not await grp_svc.is_group_admin(group_id, current_user.id, db):
        raise HTTPException(status_code=403, detail="Only the group owner can share the invite link.")
    # Backfill a token for legacy groups that predate the column.
    if not group.invite_token:
        group.invite_token = secrets.token_urlsafe(24)
        await db.commit()
    return {
        "success": True,
        "token": group.invite_token,
        "url": f"https://www.extendedmindsai.com/join/{group.invite_token}",
        "name": group.name,
    }


@app.post("/api/groups/join/{token}")
async def join_group(
    token: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Join a group via its shareable invite link. No Pro requirement."""
    result = await grp_svc.join_group_by_token(token, current_user, db)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.delete("/api/groups/{group_id}/members/{user_id}")
async def remove_group_member(
    group_id: int,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a member from a group. Only the group admin can do this. Admin cannot remove themselves."""
    admin = await db.scalar(
        select(GroupMember).where(
            GroupMember.group_id == group_id,
            GroupMember.user_id == current_user.id,
            GroupMember.role == "admin",
        )
    )
    if not admin:
        raise HTTPException(status_code=403, detail="Only the group admin can remove members")
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Admin cannot remove themselves. Delete the group instead.")
    row = await db.scalar(
        select(GroupMember).where(
            GroupMember.group_id == group_id,
            GroupMember.user_id == user_id,
        )
    )
    if not row:
        raise HTTPException(status_code=404, detail="Member not found in this group")
    await db.delete(row)
    await db.commit()
    return {"success": True}


@app.delete("/api/groups/{group_id}/leave")
async def leave_group(
    group_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    row = await db.scalar(
        select(GroupMember).where(
            GroupMember.group_id == group_id,
            GroupMember.user_id == current_user.id,
        )
    )
    if row:
        await db.delete(row)
        await db.commit()
    # Clear active group if it was this one
    if current_user.active_group_id == group_id:
        await grp_svc.set_active_group(current_user.id, None, db)
    return {"success": True}


@app.delete("/api/groups/{group_id}")
async def delete_group(
    group_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a group entirely. Only the group admin (creator) can do this."""
    member = await db.scalar(
        select(GroupMember).where(
            GroupMember.group_id == group_id,
            GroupMember.user_id == current_user.id,
            GroupMember.role == "admin",
        )
    )
    if not member:
        raise HTTPException(status_code=403, detail="Only the group admin can delete this group")

    # Delete members, messages, then the group; preserve annotations (just unlink message_id)
    await db.execute(sql_delete(GroupMember).where(GroupMember.group_id == group_id))
    group_msg_ids = (
        await db.execute(select(Message.id).where(Message.group_id == group_id))
    ).scalars().all()
    if group_msg_ids:
        await db.execute(update(LabelAnnotation).where(LabelAnnotation.message_id.in_(group_msg_ids)).values(message_id=None))
    await db.execute(sql_delete(Message).where(Message.group_id == group_id))
    group = await db.get(Group, group_id)
    if group:
        await db.delete(group)
    await db.commit()
    return {"success": True}


@app.post("/api/groups/{group_id}/settle")
async def settle_group_expenses(
    group_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Record a settlement marker for a group — resets the shared balance baseline."""
    member = await db.scalar(
        select(GroupMember).where(
            GroupMember.group_id == group_id,
            GroupMember.user_id == current_user.id,
        )
    )
    if not member:
        raise HTTPException(status_code=403, detail="Not a member of this group")

    now = datetime.utcnow()
    marker = Message(
        user_id=current_user.id,
        group_id=group_id,
        content="Expenses settled",
        message_type=MessageType.TEXT,
        tags={
            "primary_bucket": "Track",
            "settlement_marker": True,
            "settled_by_name": current_user.name,
            "settled_by_id": current_user.id,
        },
        created_at=now,
    )
    db.add(marker)
    await db.commit()
    await db.refresh(marker)

    # Bust bootstrap cache for all group members so they see the settlement immediately
    from services import redis_cache as _rc_settle
    settle_members = await grp_svc.get_group_members(group_id, db)
    for _sm in settle_members:
        asyncio.create_task(_rc_settle.cache_del(_rc_settle.bootstrap_key(_sm["id"], group_id)))

    return {"success": True, "settled_at": now.isoformat(), "message_id": marker.id}


@app.delete("/api/pro/members/{phone_number}")
async def remove_pro_member(
    phone_number: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a member from the owner's Pro account."""
    acct = await grp_svc.get_pro_account_for_user(current_user.id, db)
    if not acct:
        raise HTTPException(status_code=403, detail="Pro account not found")
    member = await db.scalar(
        select(ProAccountMember).where(
            ProAccountMember.account_id == acct.id,
            ProAccountMember.phone_number == phone_number,
        )
    )
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    await db.delete(member)
    await db.commit()
    return {"success": True}


# ── Coupon: validate (public, no auth needed for preview) ─────────────
class CouponValidateRequest(BaseModel):
    code: str

class CouponRedeemRequest(BaseModel):
    code: str

class CouponCreateRequest(BaseModel):
    code: Optional[str] = None
    description: Optional[str] = None
    discount_type: str = "free"      # free | percent | fixed
    discount_value: int = 100
    duration_days: Optional[int] = 30
    max_uses: Optional[int] = None
    expires_in_days: Optional[int] = None


@app.post("/api/pro/validate-coupon")
async def validate_coupon(
    req: CouponValidateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await cpn_svc.validate(req.code, current_user, db)
    return result


@app.post("/api/pro/redeem-coupon")
async def redeem_coupon(
    req: CouponRedeemRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await cpn_svc.redeem(req.code, current_user, db)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.get("/api/promo/founding")
async def founding_promo(db: AsyncSession = Depends(get_db)):
    """Public (no auth) — powers the 'X of N founding spots left' pill on web + iOS.
    Returns live FOUNDER-coupon usage. `enabled` is the display flag
    (FOUNDING_PILL_ENABLED); `active` means the offer is still claimable.
    The UI shows the pill only when enabled && active, so it stays invisible
    until you flip the flag for public launch."""
    return await cpn_svc.founding_status(db)


# ── Product analytics (first-party) ─────────────────────────────────────
class EventIn(BaseModel):
    event:       str
    props:       Optional[dict] = None
    session_id:  Optional[str]  = None
    anon_id:     Optional[str]  = None
    ts:          Optional[str]  = None   # client ISO8601 timestamp
    platform:    Optional[str]  = None
    app_version: Optional[str]  = None


class EventBatchIn(BaseModel):
    events: List[EventIn]


@app.post("/api/events")
async def ingest_events(batch: EventBatchIn, request: Request, db: AsyncSession = Depends(get_db)):
    """Public batched analytics ingest. Works logged-in (user_id resolved from
    the bearer token) or anonymous (correlated by anon_id). Fire-and-forget."""
    auth_hdr = request.headers.get("Authorization", "")
    token    = auth_hdr[7:] if auth_hdr.lower().startswith("bearer ") else None
    user_id  = decode_user_id_safe(token)
    rows = []
    for e in (batch.events or [])[:200]:
        client_ts = None
        if e.ts:
            try:
                client_ts = datetime.fromisoformat(e.ts.replace("Z", "+00:00")).replace(tzinfo=None)
            except Exception:
                client_ts = None
        rows.append(AnalyticsEvent(
            user_id=user_id,
            anon_id=(e.anon_id or "")[:64] or None,
            session_id=(e.session_id or "")[:64] or None,
            event=(e.event or "")[:60],
            props=e.props,
            platform=(e.platform or "")[:20] or None,
            app_version=(e.app_version or "")[:20] or None,
            client_ts=client_ts,
        ))
    if rows:
        db.add_all(rows)
        await db.commit()
    return {"ok": True, "stored": len(rows)}


# ── Admin: coupon management ────────────────────────────────────────────
@app.post("/api/admin/coupons")
async def admin_create_coupon(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    admin_secret = os.getenv("ADMIN_SECRET", "")
    provided = request.headers.get("X-Admin-Secret") or request.query_params.get("admin_secret") or ""
    if not admin_secret or provided != admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden")
    body = await request.json()
    req  = CouponCreateRequest(**body)
    result = await cpn_svc.create_coupon(
        code=req.code, description=req.description,
        discount_type=req.discount_type, discount_value=req.discount_value,
        duration_days=req.duration_days, max_uses=req.max_uses,
        expires_in_days=req.expires_in_days, db=db,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.get("/api/admin/coupons")
async def admin_list_coupons(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    admin_secret = os.getenv("ADMIN_SECRET", "")
    provided = request.headers.get("X-Admin-Secret") or request.query_params.get("admin_secret") or ""
    if not admin_secret or provided != admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden")
    coupons = await cpn_svc.list_coupons(db)
    return {"coupons": coupons}


@app.patch("/api/admin/coupons/{coupon_id}")
async def admin_update_coupon(
    coupon_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    admin_secret = os.getenv("ADMIN_SECRET", "")
    provided = request.headers.get("X-Admin-Secret") or request.query_params.get("admin_secret") or ""
    if not admin_secret or provided != admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden")
    body = await request.json()
    result = await db.execute(select(CouponCode).where(CouponCode.id == coupon_id))
    coupon = result.scalar_one_or_none()
    if not coupon:
        raise HTTPException(status_code=404, detail="Coupon not found")
    if "max_uses" in body:
        coupon.max_uses = int(body["max_uses"]) if body["max_uses"] is not None else None
    await db.commit()
    return {"success": True, "code": coupon.code, "max_uses": coupon.max_uses}


# ── Payments (Razorpay) ─────────────────────────────────────────────────────

class PaymentCreateOrderRequest(BaseModel):
    plan: str  # "monthly" | "annual"

class PaymentVerifyRequest(BaseModel):
    order_id: str
    payment_id: str
    signature: str


@app.post("/api/payments/create-order")
async def create_payment_order(
    req: PaymentCreateOrderRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await pay_svc.create_order(current_user, req.plan, db)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/api/payments/verify")
async def verify_payment(
    req: PaymentVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await pay_svc.verify_and_activate(
        current_user, req.order_id, req.payment_id, req.signature, db
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/webhook/razorpay")
async def razorpay_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    body = await request.body()
    signature = request.headers.get("X-Razorpay-Signature", "")
    ok = await pay_svc.handle_webhook(body, signature, db)
    return {"status": "ok" if ok else "ignored"}


# ── Apple In-App Purchase ────────────────────────────────────────────────────

class IAPVerifyRequest(BaseModel):
    transaction_id: str
    original_transaction_id: str


@app.post("/api/iap/verify")
async def iap_verify(
    req: IAPVerifyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Called by iOS immediately after a successful StoreKit 2 purchase.
    Verifies the transaction with Apple's App Store Server API and activates Pro.
    The iOS client must call transaction.finish() AFTER this returns success.
    """
    result = await iap_service.verify_and_activate(
        current_user, req.transaction_id, req.original_transaction_id, db
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result


@app.post("/webhook/apple")
async def apple_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """
    App Store Server Notifications V2.
    Apple sends renewal, expiration, refund, and billing-retry events here.
    Must return HTTP 200 quickly — Apple retries on failure (up to 3 times).
    Configure in App Store Connect → your app → Subscriptions → Server URL.
    """
    body = await request.json()
    signed_payload = body.get("signedPayload", "")
    if not signed_payload:
        # Malformed — acknowledge so Apple stops retrying this exact payload
        return {"status": "ignored"}

    ok = await iap_service.handle_notification(signed_payload, db)
    # Return 200 regardless so Apple doesn't flood us with retries on transient issues.
    # Genuine signature failures are logged; benign unknowns are silently acked.
    return {"status": "ok" if ok else "signature_error"}


@app.delete("/api/admin/coupons/{coupon_id}")
async def admin_deactivate_coupon(
    coupon_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    admin_secret = os.getenv("ADMIN_SECRET", "")
    provided = request.headers.get("X-Admin-Secret") or request.query_params.get("admin_secret") or ""
    if not admin_secret or provided != admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden")
    ok = await cpn_svc.deactivate(coupon_id, db)
    return {"success": ok}


# ── Admin: grant Pro ────────────────────────────────────────────────
@app.post("/api/admin/grant-pro")
async def admin_grant_pro(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    admin_secret = os.getenv("ADMIN_SECRET", "")
    provided = request.headers.get("X-Admin-Secret") or request.query_params.get("admin_secret") or ""
    if not admin_secret or provided != admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden")
    body = await request.json()
    phone = body.get("phone_number")
    user  = await db.scalar(select(User).where(User.phone_number == phone))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_pro = True
    # Create Pro account if not exists
    acct = await grp_svc.get_or_create_pro_account(user, db)
    await db.commit()
    return {"success": True, "message": f"Pro granted to {user.name}", "account_id": acct.id}


# ── Admin: product analytics readouts ───────────────────────────────────
def _check_admin(request: Request):
    admin_secret = os.getenv("ADMIN_SECRET", "")
    provided = request.headers.get("X-Admin-Secret") or request.query_params.get("admin_secret") or ""
    if not admin_secret or provided != admin_secret:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/api/admin/funnel")
async def admin_funnel(request: Request, db: AsyncSession = Depends(get_db)):
    """Activation funnel — distinct users reaching each step. Shows where people drop off."""
    _check_admin(request)

    async def users_with(event: str) -> int:
        return await db.scalar(
            select(func.count(func.distinct(AnalyticsEvent.user_id)))
            .where(AnalyticsEvent.event == event, AnalyticsEvent.user_id.isnot(None))
        ) or 0

    async def users_with_at_least(event: str, n: int) -> int:
        sub = (
            select(AnalyticsEvent.user_id)
            .where(AnalyticsEvent.event == event, AnalyticsEvent.user_id.isnot(None))
            .group_by(AnalyticsEvent.user_id)
            .having(func.count() >= n)
            .subquery()
        )
        return await db.scalar(select(func.count()).select_from(sub)) or 0

    total_users   = await db.scalar(select(func.count(User.id))) or 0
    signed_up     = await users_with("signup_completed") or await users_with("login_completed")
    steps = [
        {"step": "Signed up",           "users": signed_up},
        {"step": "First capture",       "users": await users_with("capture_sent")},
        {"step": "Captured again (≥2)", "users": await users_with_at_least("capture_sent", 2)},
        {"step": "Searched",            "users": await users_with("search_performed")},
        {"step": "Viewed Pro",          "users": await users_with("paywall_viewed")},
        {"step": "Created a group",     "users": await users_with("group_created")},
        {"step": "Became Pro",          "users": await users_with("pro_activated")},
    ]
    zero_searches = await db.scalar(
        select(func.count()).select_from(AnalyticsEvent)
        .where(AnalyticsEvent.event == "search_performed",
               AnalyticsEvent.props["results"].astext == "0")
    ) or 0
    return {"total_users": total_users, "steps": steps,
            "signals": {"zero_result_searches": zero_searches}}


@app.get("/api/admin/events")
async def admin_events(request: Request, user_id: Optional[int] = None,
                       limit: int = 200, db: AsyncSession = Depends(get_db)):
    """Recent event stream (optionally one user's journey) for eyeballing friction."""
    _check_admin(request)
    q = select(AnalyticsEvent).order_by(AnalyticsEvent.id.desc()).limit(min(limit, 1000))
    if user_id is not None:
        q = q.where(AnalyticsEvent.user_id == user_id)
    rows = (await db.execute(q)).scalars().all()
    return [
        {"id": r.id, "user_id": r.user_id, "anon_id": r.anon_id, "event": r.event,
         "props": r.props, "platform": r.platform, "session_id": r.session_id,
         "ts": (r.client_ts or r.created_at).isoformat()}
        for r in rows
    ]


@app.post("/api/admin/reembed")
async def admin_reembed(request: Request, user_id: Optional[int] = None,
                        limit: int = 50, db: AsyncSession = Depends(get_db)):
    """Re-generate embeddings for messages that have NULL embedding.

    Useful when a transient Gemini error caused embedding generation to fail
    silently at capture time, leaving the message invisible to semantic search.
    Pass ?user_id=X to scope to one user; omit to run across all users (capped at limit).
    """
    _check_admin(request)
    from services.embedding_service import embedding_service

    q = select(Message).where(Message.embedding.is_(None)).order_by(Message.created_at.desc()).limit(min(limit, 200))
    if user_id is not None:
        q = q.where(Message.user_id == user_id)
    rows = (await db.execute(q)).scalars().all()

    ok, failed = 0, 0
    for msg in rows:
        try:
            text_to_embed = (msg.content or "").strip()
            if not text_to_embed:
                continue
            tags = msg.tags if isinstance(msg.tags, dict) else {}
            enriched = " ".join(filter(None, [
                text_to_embed,
                tags.get("essence", "") or (msg.summary or ""),
                " ".join(tags.get("keywords", [])),
            ])).strip()
            embedding = await embedding_service.aembed(enriched, task_type="RETRIEVAL_DOCUMENT")
            await db.execute(
                update(Message).where(Message.id == msg.id).values(embedding=embedding)
            )
            ok += 1
        except Exception as e:
            print(f"[reembed] msg {msg.id} failed: {e}")
            failed += 1
    await db.commit()
    return {"reembedded": ok, "failed": failed, "total_found": len(rows)}
