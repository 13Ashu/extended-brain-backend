"""
Reminder Service
────────────────────────────────────────────────────────────────────────────
Handles creation, scheduling, and delivery of reminders via Telegram.
Reminders are stored as messages with type "reminder" in the DB, plus a
dedicated reminders table for scheduling state.

Flow:
  1. MessageProcessor detects reminder intent → calls ReminderService.create()
  2. ReminderService stores reminder + schedules it
  3. Background scheduler loop fires due reminders → sends Telegram message
  4. Reminder marked as sent / snoozed

Save as: services/reminder_service.py
"""

from __future__ import annotations
from database import Base, User, Message, Category, MessageType, async_session_maker
import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import select, update, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Text, DateTime, ForeignKey, Boolean, Integer, JSON
from sqlalchemy.ext.asyncio import AsyncSession

from database import Base, User, Message, Category, MessageType, async_session_maker
from cerebras_client import CerebrasClient


# ─────────────────────────────────────────────────────────────────────────────
# DB Model
# ─────────────────────────────────────────────────────────────────────────────

class Reminder(Base):
    __tablename__ = "reminders"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    message_id: Mapped[Optional[int]] = mapped_column(ForeignKey("messages.id"), nullable=True)

    # What to remind
    content: Mapped[str] = mapped_column(Text)           # original user text
    task: Mapped[str] = mapped_column(Text)              # cleaned task label
    
    # When to fire
    remind_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")

    # Delivery
    telegram_chat_id: Mapped[Optional[str]] = mapped_column(String(50))
    
    # State
    is_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    is_cancelled: Mapped[bool] = mapped_column(Boolean, default=False)
    snooze_count: Mapped[int] = mapped_column(Integer, default=0)
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Recurrence (future)
    recurrence: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Reminder {self.id} — {self.task} at {self.remind_at}>"


# ─────────────────────────────────────────────────────────────────────────────
# Telegram sender (thin wrapper — no extra deps)
# ─────────────────────────────────────────────────────────────────────────────

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"

async def send_telegram(chat_id: str, text: str, token: str, parse_mode: str = "Markdown") -> bool:
    url = TELEGRAM_API.format(token=token)
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
        self.cerebras = cerebras_client
        self.telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._scheduler_running = False

    # ──────────────────────────────────────────────────────────────
    # Create a reminder (called from MessageProcessor)
    # ──────────────────────────────────────────────────────────────

    async def create(
        self,
        user: User,
        content: str,
        analysis: Dict,               # full LLM analysis from MessageProcessor
        message_id: Optional[int],
        db: AsyncSession,
    ) -> Optional[Reminder]:
        """
        Parse remind_at from analysis and persist a Reminder row.
        Returns None if no valid time could be extracted.
        """
        remind_at = self._resolve_remind_at(analysis, user.timezone)
        if not remind_at:
            print(f"[reminder] Could not resolve time from analysis: {analysis.get('due_date')}, {analysis.get('event_time')}")
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
    # Scheduler loop — run as background task in main.py
    # ──────────────────────────────────────────────────────────────

    async def run_scheduler(self, poll_interval: int = 30):
        """
        Long-running coroutine. Call once at app startup:
            asyncio.create_task(reminder_service.run_scheduler())
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
        if not reminder.telegram_chat_id or not self.telegram_token:
            print(f"[reminder] No telegram config for reminder #{reminder.id}")
            return

        # Format the message
        local_time = self._to_local_time(reminder.remind_at, reminder.timezone)
        text = (
            f"⏰ *Reminder*\n\n"
            f"{reminder.task}\n\n"
            f"_Scheduled for {local_time}_"
        )

        success = await send_telegram(
            chat_id=reminder.telegram_chat_id,
            text=text,
            token=self.telegram_token,
        )

        if success:
            await db.execute(
                update(Reminder)
                .where(Reminder.id == reminder.id)
                .values(is_sent=True, sent_at=datetime.utcnow())
            )
            await db.commit()
            print(f"[reminder] ✅ Sent #{reminder.id} — {reminder.task}")
        else:
            print(f"[reminder] ❌ Failed to send #{reminder.id}")

    # ──────────────────────────────────────────────────────────────
    # Snooze / Cancel (called from webhook when user replies)
    # ──────────────────────────────────────────────────────────────

    async def snooze(
        self,
        reminder_id: int,
        minutes: int,
        db: AsyncSession,
    ) -> Optional[Reminder]:
        result = await db.execute(select(Reminder).where(Reminder.id == reminder_id))
        reminder = result.scalar_one_or_none()
        if not reminder:
            return None

        new_time = datetime.utcnow() + timedelta(minutes=minutes)
        reminder.remind_at = new_time
        reminder.is_sent = False
        reminder.snooze_count += 1
        await db.commit()
        await db.refresh(reminder)
        return reminder

    async def cancel(self, reminder_id: int, db: AsyncSession) -> bool:
        result = await db.execute(select(Reminder).where(Reminder.id == reminder_id))
        reminder = result.scalar_one_or_none()
        if not reminder:
            return False
        reminder.is_cancelled = True
        await db.commit()
        return True

    async def list_upcoming(
        self,
        user_id: int,
        db: AsyncSession,
        limit: int = 10,
    ) -> List[Reminder]:
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
        """
        Build a UTC datetime from LLM analysis fields.
        Priority: due_date + event_time > due_date alone > today + event_time
        """
        due_date_str = analysis.get("due_date")       # "YYYY-MM-DD"
        event_time   = analysis.get("event_time")     # "HH:MM" or None

        if not due_date_str and not event_time:
            return None

        tz = ZoneInfo(user_tz or "UTC")

        # Base date
        if due_date_str:
            try:
                base = datetime.fromisoformat(due_date_str)
            except ValueError:
                base = datetime.utcnow().replace(tzinfo=ZoneInfo("UTC"))
        else:
            base = datetime.now(tz)

        # Apply time
        if event_time:
            try:
                h, m = map(int, event_time.split(":"))
                local_dt = base.replace(hour=h, minute=m, second=0, microsecond=0, tzinfo=tz)
            except Exception:
                local_dt = base.replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=tz)
        else:
            # Default to 9am local if only a date was given
            local_dt = base.replace(hour=9, minute=0, second=0, microsecond=0, tzinfo=tz)

        # Convert to UTC
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
            tz = ZoneInfo(tz_str or "UTC")
            local = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)
            return local.strftime("%a, %d %b %Y at %I:%M %p")
        except Exception:
            return dt.strftime("%Y-%m-%d %H:%M UTC")