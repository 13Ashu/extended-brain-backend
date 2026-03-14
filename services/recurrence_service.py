"""
Recurrence Service
─────────────────────────────────────────────────────────────
• Detects recurring task intent from natural language
• Stores recurrence rules in the recurrences table
• Scheduler fires due recurrences, auto-creates next instance
• Supports: daily, weekly (day), monthly (date)
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import and_, select, update, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Boolean, Integer, DateTime, ForeignKey, Text

from database import async_session_maker, Base, User, Message, Category
from cerebras_client import CerebrasClient


IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")


# ── ORM Model ─────────────────────────────────────────────────────────────────

class Recurrence(Base):
    __tablename__ = "recurrences"

    id:               Mapped[int]           = mapped_column(primary_key=True)
    user_id:          Mapped[int]           = mapped_column(ForeignKey("users.id"))
    message_id:       Mapped[Optional[int]] = mapped_column(ForeignKey("messages.id"), nullable=True)
    rule:             Mapped[str]           = mapped_column(String(50))   # daily/weekly/monthly
    day_of_week:      Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 0=Mon
    time_of_day:      Mapped[str]           = mapped_column(String(5))    # HH:MM IST
    last_fired:       Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    next_fire:        Mapped[datetime]      = mapped_column(DateTime)
    is_active:        Mapped[bool]          = mapped_column(Boolean, default=True)
    template_content: Mapped[str]           = mapped_column(Text)
    created_at:       Mapped[datetime]      = mapped_column(DateTime, default=datetime.utcnow)


# ── Service ───────────────────────────────────────────────────────────────────

RECURRENCE_SIGNALS = {
    "every day", "daily", "every morning", "every evening", "every night",
    "every monday", "every tuesday", "every wednesday", "every thursday",
    "every friday", "every saturday", "every sunday",
    "every week", "weekly", "every month", "monthly",
    "each day", "each week", "each month",
    "repeat", "recurring", "remind me every",
}

DAY_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


class RecurrenceService:

    def __init__(self, cerebras_client: CerebrasClient):
        self.cerebras = cerebras_client

    # ──────────────────────────────────────────────────────────────
    # Detect recurring intent
    # ──────────────────────────────────────────────────────────────

    def is_recurring(self, content: str) -> bool:
        lc = content.lower()
        return any(sig in lc for sig in RECURRENCE_SIGNALS)

    async def parse_recurrence(self, content: str) -> Optional[Dict]:
        """
        Parse recurrence rule from natural language.
        Returns {rule, day_of_week, time_of_day, task} or None.
        """
        prompt = f"""Parse a recurring task from natural language.

INPUT: "{content}"

RULES:
- rule: "daily" | "weekly" | "monthly"
- day_of_week: 0=Monday, 1=Tuesday ... 6=Sunday (only for weekly)
- time_of_day: "HH:MM" in 24hr format (default "09:00" if not specified)
- task: the actual task content without the recurrence part

Return ONLY this JSON:
{{
  "rule":         "daily | weekly | monthly",
  "day_of_week":  null,
  "time_of_day":  "HH:MM",
  "task":         "clean task description"
}}

Examples:
"remind me every monday to do weekly review" →
  {{"rule":"weekly","day_of_week":0,"time_of_day":"09:00","task":"do weekly review"}}

"drink water every day at 8am" →
  {{"rule":"daily","day_of_week":null,"time_of_day":"08:00","task":"drink water"}}"""

        try:
            response = await self.cerebras.chat_lite(prompt, max_tokens=200)
            if response.get("rule") and response.get("task"):
                return response
        except Exception as e:
            print(f"[recurrence] Parse failed: {e}")

        return None

    # ──────────────────────────────────────────────────────────────
    # Create recurrence
    # ──────────────────────────────────────────────────────────────

    async def create(
        self,
        user_id: int,
        parsed: Dict,
        original_content: str,
        db: AsyncSession,
    ) -> Optional[Recurrence]:
        try:
            next_fire = self._compute_next_fire(
                rule=parsed["rule"],
                day_of_week=parsed.get("day_of_week"),
                time_of_day=parsed.get("time_of_day", "09:00"),
            )

            rec = Recurrence(
                user_id=user_id,
                rule=parsed["rule"],
                day_of_week=parsed.get("day_of_week"),
                time_of_day=parsed.get("time_of_day", "09:00"),
                next_fire=next_fire,
                template_content=parsed.get("task", original_content),
                is_active=True,
            )
            db.add(rec)
            await db.commit()
            await db.refresh(rec)
            return rec
        except Exception as e:
            print(f"[recurrence] Create failed: {e}")
            return None

    # ──────────────────────────────────────────────────────────────
    # Scheduler — fires due recurrences
    # ──────────────────────────────────────────────────────────────

    async def run(self):
        """Called every minute from scheduler."""
        now_utc = datetime.utcnow()

        async with async_session_maker() as session:
            result = await session.execute(
                select(Recurrence, User)
                .join(User, Recurrence.user_id == User.id)
                .where(
                    and_(
                        Recurrence.is_active == True,
                        Recurrence.next_fire <= now_utc,
                        User.telegram_chat_id.isnot(None),
                    )
                )
            )
            due = result.all()

        for rec, user in due:
            try:
                await self._fire(rec, user)
            except Exception as e:
                print(f"[recurrence] Fire failed for rec {rec.id}: {e}")

    async def _fire(self, rec: Recurrence, user: User):
        """Create a new todo message and send notification."""
        from database import Category as Cat

        async with async_session_maker() as session:
            # Get or create To-Do category
            cat = await session.scalar(
                select(Cat).where(
                    and_(Cat.user_id == user.id, Cat.name == "To-Do")
                )
            )
            if not cat:
                cat = Cat(user_id=user.id, name="To-Do", description="Tasks")
                session.add(cat)
                await session.flush()

            now_ist = datetime.now(IST)
            today   = now_ist.strftime("%Y-%m-%d")

            # Create the recurring todo
            msg = Message(
                user_id=user.id,
                category_id=cat.id,
                content=rec.template_content,
                message_type=__import__("models").MessageType("text"),
                summary=rec.template_content[:100],
                tags={
                    "all_buckets":    ["To-Do"],
                    "primary_bucket": "To-Do",
                    "due_date":       today,
                    "priority":       "normal",
                    "recurring":      True,
                    "recurrence_id":  rec.id,
                },
            )
            session.add(msg)

            # Compute next fire
            next_fire = self._compute_next_fire(
                rule=rec.rule,
                day_of_week=rec.day_of_week,
                time_of_day=rec.time_of_day,
                after=datetime.utcnow(),
            )

            # Update recurrence
            await session.execute(
                update(Recurrence)
                .where(Recurrence.id == rec.id)
                .values(last_fired=datetime.utcnow(), next_fire=next_fire)
            )
            await session.commit()

        # Notify user
        rule_str = self._rule_display(rec)
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={
                    "chat_id":    user.telegram_chat_id,
                    "text":       (
                        f"🔁 *Recurring task*\n\n"
                        f"{rec.template_content}\n\n"
                        f"_{rule_str}_"
                    ),
                    "parse_mode": "Markdown",
                    "reply_markup": {
                        "inline_keyboard": [[
                            {"text": "✅ Done",   "callback_data": f"done:{msg.id}"},
                            {"text": "⏸ Pause",  "callback_data": f"pause_rec:{rec.id}"},
                        ]]
                    }
                },
            )

        print(f"[recurrence] Fired rec {rec.id} for user {user.id}")

    # ──────────────────────────────────────────────────────────────
    # Pause / resume
    # ──────────────────────────────────────────────────────────────

    async def pause(self, rec_id: int) -> bool:
        try:
            async with async_session_maker() as session:
                await session.execute(
                    update(Recurrence)
                    .where(Recurrence.id == rec_id)
                    .values(is_active=False)
                )
                await session.commit()
            return True
        except Exception as e:
            print(f"[recurrence] Pause failed: {e}")
            return False

    async def resume(self, rec_id: int) -> bool:
        try:
            async with async_session_maker() as session:
                rec = await session.scalar(
                    select(Recurrence).where(Recurrence.id == rec_id)
                )
                if not rec:
                    return False
                next_fire = self._compute_next_fire(
                    rec.rule, rec.day_of_week, rec.time_of_day
                )
                await session.execute(
                    update(Recurrence)
                    .where(Recurrence.id == rec_id)
                    .values(is_active=True, next_fire=next_fire)
                )
                await session.commit()
            return True
        except Exception as e:
            print(f"[recurrence] Resume failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _compute_next_fire(
        self,
        rule: str,
        day_of_week: Optional[int],
        time_of_day: str,
        after: Optional[datetime] = None,
    ) -> datetime:
        """Compute next UTC fire time from IST time_of_day."""
        after    = after or datetime.utcnow()
        now_ist  = after.replace(tzinfo=UTC).astimezone(IST)
        h, m     = map(int, time_of_day.split(":"))

        if rule == "daily":
            candidate = now_ist.replace(hour=h, minute=m, second=0, microsecond=0)
            if candidate <= now_ist:
                candidate += timedelta(days=1)

        elif rule == "weekly" and day_of_week is not None:
            days_ahead = (day_of_week - now_ist.weekday()) % 7
            if days_ahead == 0:
                candidate = now_ist.replace(hour=h, minute=m, second=0, microsecond=0)
                if candidate <= now_ist:
                    days_ahead = 7
                    candidate  = now_ist + timedelta(days=days_ahead)
                    candidate  = candidate.replace(hour=h, minute=m, second=0, microsecond=0)
            else:
                candidate = now_ist + timedelta(days=days_ahead)
                candidate = candidate.replace(hour=h, minute=m, second=0, microsecond=0)

        elif rule == "monthly":
            candidate = now_ist.replace(hour=h, minute=m, second=0, microsecond=0)
            if candidate <= now_ist:
                # Next month same day
                if now_ist.month == 12:
                    candidate = candidate.replace(year=now_ist.year + 1, month=1)
                else:
                    candidate = candidate.replace(month=now_ist.month + 1)
        else:
            candidate = now_ist + timedelta(days=1)
            candidate = candidate.replace(hour=h, minute=m, second=0, microsecond=0)

        return candidate.astimezone(UTC).replace(tzinfo=None)

    def _rule_display(self, rec: Recurrence) -> str:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if rec.rule == "daily":
            return f"Repeats daily at {rec.time_of_day} IST"
        elif rec.rule == "weekly" and rec.day_of_week is not None:
            return f"Repeats every {days[rec.day_of_week]} at {rec.time_of_day} IST"
        elif rec.rule == "monthly":
            return f"Repeats monthly at {rec.time_of_day} IST"
        return "Recurring"

    async def list_user_recurrences(
        self, user_id: int, db: AsyncSession
    ) -> List[Recurrence]:
        result = await db.execute(
            select(Recurrence)
            .where(
                and_(
                    Recurrence.user_id == user_id,
                    Recurrence.is_active == True,
                )
            )
            .order_by(Recurrence.created_at.desc())
        )
        return result.scalars().all()
