"""
Reminder Service
────────────────────────────────────────────────────────────────────────────
Handles creation, scheduling, and delivery of reminders via Telegram + APNs.
"""

from __future__ import annotations
from database import Base, User, Message, Category, MessageType, async_session_maker
import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from zoneinfo import ZoneInfo

import httpx
from jose import jwt as jose_jwt
from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, ForeignKey, Boolean, Integer, JSON

from database import Base, User, Message, Category, MessageType, async_session_maker, DeviceToken
from cerebras_client import CerebrasClient


# ─────────────────────────────────────────────────────────────────────────────
# APNs helpers
# ─────────────────────────────────────────────────────────────────────────────

_apns_jwt_cache: dict = {}   # {"token": str, "exp": float}


def _normalize_apns_key(raw: str) -> str:
    """
    Normalize a .p8 APNs private key for reliable PEM loading on all platforms.
    Handles: missing PEM headers (raw base64 only), escaped \\n sequences from
    Railway/Heroku dashboards, Windows line endings, and improperly wrapped base64.
    """
    # Unescape literal \n / \r sequences (Railway env var UI encoding)
    key = raw.replace("\\n", "\n").replace("\\r", "")
    key = key.replace("\r\n", "\n").replace("\r", "\n").strip()

    if "-----BEGIN PRIVATE KEY-----" in key:
        # Has headers — extract body lines, strip whitespace, re-wrap at 64 chars
        lines = [l.strip() for l in key.splitlines() if l.strip()]
        body = "".join(l for l in lines if not l.startswith("-----"))
    else:
        # Raw base64 only (no headers) — treat entire content as the body
        body = "".join(key.split())

    wrapped = "\n".join(body[i:i + 64] for i in range(0, len(body), 64))
    return f"-----BEGIN PRIVATE KEY-----\n{wrapped}\n-----END PRIVATE KEY-----\n"


def _build_apns_jwt() -> str:
    """Build (or return cached) APNs provider JWT — valid 1 h, refresh at 55 min."""
    global _apns_jwt_cache
    now = time.time()
    if _apns_jwt_cache.get("exp", 0) > now + 300:
        return _apns_jwt_cache["token"]

    key_id   = os.getenv("APNS_KEY_ID", "")
    team_id  = os.getenv("APNS_TEAM_ID", "")
    auth_key = _normalize_apns_key(os.getenv("APNS_AUTH_KEY", ""))

    token = jose_jwt.encode(
        {"iss": team_id, "iat": int(now)},
        auth_key,
        algorithm="ES256",
        headers={"kid": key_id},
    )
    _apns_jwt_cache = {"token": token, "exp": now + 3300}
    return token


async def send_apns_notification(
    device_token: str,
    title: str,
    body: str,
    badge: int = 1,
    data: dict | None = None,
    category: str | None = None,
) -> bool:
    """Send a push notification to one iOS device via APNs HTTP/2."""
    key_id    = os.getenv("APNS_KEY_ID", "")
    team_id   = os.getenv("APNS_TEAM_ID", "")
    auth_key  = os.getenv("APNS_AUTH_KEY", "")
    bundle_id = os.getenv("APNS_BUNDLE_ID", "")

    if not all([key_id, team_id, auth_key, bundle_id, device_token]):
        print("[apns] Missing config — set APNS_KEY_ID / APNS_TEAM_ID / APNS_AUTH_KEY / APNS_BUNDLE_ID")
        return False

    production = os.getenv("APNS_PRODUCTION", "false").lower() == "true"
    host = "api.push.apple.com" if production else "api.sandbox.push.apple.com"
    url  = f"https://{host}/3/device/{device_token}"

    headers = {
        "authorization":  f"bearer {_build_apns_jwt()}",
        "apns-topic":     bundle_id,
        "apns-push-type": "alert",
        "apns-priority":  "10",
    }
    aps: dict = {
        "alert": {"title": title, "body": body},
        "badge": badge,
        "sound": "default",
    }
    if category:
        aps["category"] = category
    payload: dict = {"aps": aps}
    if data:
        payload.update(data)

    try:
        async with httpx.AsyncClient(http2=True, timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code == 200:
            print(f"[apns] ✅ Sent to ...{device_token[-8:]}")
            return True
        print(f"[apns] ❌ {resp.status_code}: {resp.text[:200]}")
        return False
    except Exception as e:
        print(f"[apns] Error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# DB Model
# ─────────────────────────────────────────────────────────────────────────────

class Reminder(Base):
    __tablename__ = "reminders"

    id:               Mapped[int]           = mapped_column(primary_key=True)
    user_id:          Mapped[int]           = mapped_column(ForeignKey("users.id"), index=True)
    message_id:       Mapped[Optional[int]] = mapped_column(ForeignKey("messages.id"), nullable=True)
    content:          Mapped[str]           = mapped_column(Text)
    task:             Mapped[str]           = mapped_column(Text)
    remind_at:        Mapped[datetime]      = mapped_column(DateTime, index=True)
    timezone:         Mapped[str]           = mapped_column(String(50), default="UTC")
    telegram_chat_id: Mapped[Optional[str]] = mapped_column(String(50))
    is_sent:          Mapped[bool]          = mapped_column(Boolean, default=False)
    is_cancelled:     Mapped[bool]          = mapped_column(Boolean, default=False)
    snooze_count:     Mapped[int]           = mapped_column(Integer, default=0)
    sent_at:          Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    recurrence:       Mapped[Optional[dict]]    = mapped_column(JSON, nullable=True)
    created_at:       Mapped[datetime]      = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Reminder {self.id} — {self.task} at {self.remind_at}>"


# ─────────────────────────────────────────────────────────────────────────────
# Telegram sender
# ─────────────────────────────────────────────────────────────────────────────

async def send_telegram(chat_id: str, text: str, token: str, parse_mode: str = "Markdown") -> bool:
    url     = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return True
    except Exception as e:
        print(f"[telegram] Failed to send to {chat_id}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Reminder Service
# ─────────────────────────────────────────────────────────────────────────────

class ReminderService:
    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras         = cerebras_client
        self.telegram_token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._scheduler_running = False

    # ──────────────────────────────────────────────────────────────
    # Create
    # ──────────────────────────────────────────────────────────────

    async def create(
        self,
        user: User,
        content: str,
        analysis: Dict,
        message_id: Optional[int],
        db: AsyncSession,
    ) -> Optional[Reminder]:
        remind_at = self._resolve_remind_at(analysis, user.timezone)
        if not remind_at:
            print(f"[reminder] Could not resolve time from analysis")
            return None

        task = self._best_task_label(content, analysis)

        reminder = Reminder(
            user_id=user.id,
            message_id=message_id,
            content=content,
            task=task,
            remind_at=remind_at,
            timezone=user.timezone or "UTC",
            telegram_chat_id=user.telegram_chat_id,
        )
        db.add(reminder)
        await db.commit()
        await db.refresh(reminder)
        print(f"[reminder] Created #{reminder.id} — '{task}' at {remind_at} UTC")
        return reminder

    # ──────────────────────────────────────────────────────────────
    # Scheduler
    # ──────────────────────────────────────────────────────────────

    async def run_scheduler_tick(self):
        """
        Single tick — called every 60s by the master scheduler in main.py.
        """
        try:
            await self._fire_due_reminders()
        except Exception as e:
            print(f"[reminder] Tick error: {e}")

    async def run_scheduler(self, poll_interval: int = 30):
        """
        Long-running coroutine — kept for backward compatibility.
        New deployments use run_scheduler_tick() via master scheduler.
        """
        self._scheduler_running = True
        print("[reminder] Scheduler started")
        while self._scheduler_running:
            try:
                await self._fire_due_reminders()
            except Exception as e:
                print(f"[reminder] Scheduler error: {e}")
            await asyncio.sleep(poll_interval)

    async def _fire_due_reminders(self):
        now = datetime.utcnow()
        async with async_session_maker() as db:
            try:
                result = await db.execute(
                    select(Reminder).where(
                        and_(
                            Reminder.remind_at <= now,
                            Reminder.is_sent == False,
                            Reminder.is_cancelled == False,
                        )
                    )
                )
                due = result.scalars().all()
                for reminder in due:
                    await self._send_reminder(reminder, db)
            except Exception as e:
                print(f"[reminder] DB error in scheduler: {e}")

    async def _send_reminder(self, reminder: Reminder, db: AsyncSession):
        # ── APNs push (works even without Telegram) ───────────────
        try:
            async with async_session_maker() as apns_db:
                result = await apns_db.execute(
                    select(DeviceToken).where(DeviceToken.user_id == reminder.user_id)
                )
                tokens = result.scalars().all()
            for dt in tokens:
                await send_apns_notification(
                    device_token=dt.token,
                    title="⏰ Reminder",
                    body=reminder.task,
                    data={"type": "reminder", "reminder_id": reminder.id,
                          "message_id": reminder.message_id},
                    category="REMINDER_ACTION",
                )
        except Exception as e:
            print(f"[apns] Failed for reminder #{reminder.id}: {e}")

        # ── Telegram (existing path) ──────────────────────────────
        if not reminder.telegram_chat_id or not self.telegram_token:
            print(f"[reminder] No telegram config for reminder #{reminder.id}")
            # Mark sent if we sent via APNs
            async with async_session_maker() as s:
                await s.execute(
                    update(Reminder).where(Reminder.id == reminder.id)
                    .values(is_sent=True, sent_at=datetime.utcnow())
                )
                await s.commit()
            return

        local_time = self._to_local_time(reminder.remind_at, reminder.timezone)
        priority   = getattr(reminder, "priority", "normal") or "normal"
        alarm_icon = "🚨" if priority in ("high", "urgent") else "⏰"

        text = (
            f"{alarm_icon} *Reminder*\n\n"
            f"{reminder.task}\n\n"
            f"_Scheduled for {local_time}_"
        )

        reply_markup = {
            "inline_keyboard": [[
                {
                    "text":          "✅ Done",
                    "callback_data": f"done:{reminder.message_id}",
                },
                {
                    "text":          "⏰ 30 min",
                    "callback_data": f"snooze:{reminder.message_id}:30",
                },
                {
                    "text":          "⏰ 1 hr",
                    "callback_data": f"snooze:{reminder.message_id}:60",
                },
            ]]
        }

        token = self.telegram_token
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id":              reminder.telegram_chat_id,
                    "text":                 text,
                    "parse_mode":           "Markdown",
                    "disable_notification": False,
                    "reply_markup":         reply_markup,
                },
            )
            data      = resp.json()
            tg_msg_id = data.get("result", {}).get("message_id")

            # Pin high priority reminders for extra visibility
            if priority in ("high", "urgent") and tg_msg_id:
                await client.post(
                    f"https://api.telegram.org/bot{token}/pinChatMessage",
                    json={
                        "chat_id":              reminder.telegram_chat_id,
                        "message_id":           tg_msg_id,
                        "disable_notification": False,
                    },
                )

        if data.get("ok"):
            async with async_session_maker() as session:
                await session.execute(
                    update(Reminder)
                    .where(Reminder.id == reminder.id)
                    .values(is_sent=True, sent_at=datetime.utcnow())
                )
                # Mark linked todo done only for non-recurring reminders
                if reminder.message_id and not reminder.recurrence:
                    try:
                        msg = await session.scalar(
                            select(Message).where(Message.id == reminder.message_id)
                        )
                        if msg:
                            tags = dict(msg.tags or {})
                            tags["reminded_at"] = datetime.utcnow().isoformat()
                            # Don't auto-mark done — let user tap the Done button
                            await session.execute(
                                update(Message)
                                .where(Message.id == reminder.message_id)
                                .values(tags=tags)
                            )
                    except Exception as e:
                        print(f"[reminder] Could not update todo tags: {e}")
                await session.commit()
            print(f"[reminder] ✅ Sent #{reminder.id} — {reminder.task}")
        else:
            print(f"[reminder] ❌ Failed to send #{reminder.id}: {data}")
    # ──────────────────────────────────────────────────────────────
    # Snooze / Cancel / List
    # ──────────────────────────────────────────────────────────────

    async def snooze(self, reminder_id: int, minutes: int, db: AsyncSession) -> Optional[Reminder]:
        result   = await db.execute(select(Reminder).where(Reminder.id == reminder_id))
        reminder = result.scalar_one_or_none()
        if not reminder:
            return None
        reminder.remind_at    = datetime.utcnow() + timedelta(minutes=minutes)
        reminder.is_sent      = False
        reminder.snooze_count += 1
        await db.commit()
        await db.refresh(reminder)
        return reminder

    async def cancel(self, reminder_id: int, db: AsyncSession) -> bool:
        result   = await db.execute(select(Reminder).where(Reminder.id == reminder_id))
        reminder = result.scalar_one_or_none()
        if not reminder:
            return False
        reminder.is_cancelled = True
        await db.commit()
        return True

    async def list_upcoming(self, user_id: int, db: AsyncSession, limit: int = 10) -> List[Reminder]:
        result = await db.execute(
            select(Reminder)
            .where(
                and_(
                    Reminder.user_id == user_id,
                    Reminder.is_sent == False,
                    Reminder.is_cancelled == False,
                    Reminder.remind_at >= datetime.utcnow(),
                )
            )
            .order_by(Reminder.remind_at)
            .limit(limit)
        )
        return result.scalars().all()

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _resolve_remind_at(self, analysis: Dict, user_tz: str) -> Optional[datetime]:
        due_date_str = analysis.get("due_date")
        event_time   = analysis.get("event_time")

        if not due_date_str and not event_time:
            return None

        tz = ZoneInfo(user_tz or "UTC")

        if due_date_str:
            try:
                base = datetime.fromisoformat(due_date_str)
            except ValueError:
                base = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        else:
            base = datetime.now(tz)

        if event_time:
            try:
                h, m     = map(int, event_time.split(":"))
                local_dt = base.replace(hour=h, minute=m, second=0, microsecond=0, tzinfo=tz)
            except Exception:
                local_dt = base.replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=tz)
        else:
            local_dt = base.replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=tz)

        return local_dt.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)

    def _best_task_label(self, content: str, analysis: Dict) -> str:
        actionables = analysis.get("actionables", [])
        if actionables:
            return actionables[0]
        essence = analysis.get("essence", "")
        if essence:
            return essence
        return content[:120]

    def _to_local_time(self, dt: datetime, tz_str: str) -> str:
        try:
            tz    = ZoneInfo(tz_str or "UTC")
            local = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
            return local.strftime("%a, %d %b %Y at %I:%M %p")
        except Exception:
            return dt.strftime("%Y-%m-%d %H:%M UTC")
