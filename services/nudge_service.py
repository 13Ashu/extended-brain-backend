"""
Nudge Service
─────────────────────────────────────────────────────────────
• Idle nudges  — tracks overdue todos (APNs delivery not yet implemented)
• Follow-up tracking — messages with people + action verbs tracked after 48hrs
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List
from zoneinfo import ZoneInfo

from sqlalchemy import and_, select, update, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import async_session_maker, Message, User

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

FOLLOWUP_HOURS = 48


class NudgeService:

    # ──────────────────────────────────────────────────────────────
    # Idle nudges
    # ──────────────────────────────────────────────────────────────

    async def run_idle_nudges(self):
        """
        Called once per day from scheduler.
        Finds todos created/due before today, not done, not snoozed until tomorrow.
        """
        now_utc = datetime.utcnow()
        today   = datetime.now(IST).strftime("%Y-%m-%d")

        async with async_session_maker() as session:
            users = await session.execute(
                select(User).where(User.is_active == True)
            )
            users = users.scalars().all()

        for user in users:
            try:
                await self._nudge_user(user, today, now_utc)
            except Exception as e:
                print(f"[nudge] Failed for user {user.id}: {e}")


    async def _nudge_user(self, user: User, today: str, now_utc: datetime):
        async with async_session_maker() as session:
            result = await session.execute(
                select(Message)
                .where(
                    and_(
                        Message.user_id == user.id,
                        text("messages.tags->>'due_date' <= :today"),
                        text("(messages.tags->>'done')::boolean IS NOT TRUE"),
                        text("messages.tags->'all_buckets' @> '\"To-Do\"'::jsonb"),
                        text(
                            "(messages.tags->>'snooze_until' IS NULL OR "
                            " messages.tags->>'snooze_until' <= :now)"
                        ),
                        # FIX: use SQLAlchemy column comparison for DateTime column
                        # not text() — asyncpg requires actual datetime object, not string
                        Message.created_at <= now_utc - timedelta(days=1),
                    )
                )
                .params(
                    today=today,
                    # FIX: snooze_until is stored as a JSONB string so string compare is fine
                    now=now_utc.strftime("%Y-%m-%dT%H:%M:%S"),
                )
                .limit(5)
            )
            todos = result.scalars().all()
            # APNs nudge delivery not yet implemented; todos tracked in DB only.

    # ──────────────────────────────────────────────────────────────
    # Snooze handler (called from main.py callback router)
    # ──────────────────────────────────────────────────────────────

    async def snooze_message(self, message_id: int, minutes: int) -> bool:
        """Push snooze_until forward by `minutes` from now."""
        try:
            snooze_until = (datetime.utcnow() + timedelta(minutes=minutes)).strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
            async with async_session_maker() as session:
                msg = await session.scalar(
                    select(Message).where(Message.id == message_id)
                )
                if not msg:
                    return False
                tags = dict(msg.tags or {})
                tags["snooze_until"] = snooze_until
                await session.execute(
                    update(Message)
                    .where(Message.id == message_id)
                    .values(tags=tags)
                )
                await session.commit()
            return True
        except Exception as e:
            print(f"[nudge] Snooze failed: {e}")
            return False

    # ──────────────────────────────────────────────────────────────
    # Follow-up tracking
    # ──────────────────────────────────────────────────────────────

    async def run_followup_checks(self):
        """
        Find messages that:
        - Have people in entities
        - Have actionables
        - Were saved > FOLLOWUP_HOURS ago
        - Not done, not followed up already
        Marks follow_up_sent to avoid re-processing.
        APNs delivery not yet implemented.
        """
        cutoff = datetime.utcnow() - timedelta(hours=FOLLOWUP_HOURS)

        async with async_session_maker() as session:
            result = await session.execute(
                select(Message, User)
                .join(User, Message.user_id == User.id)
                .where(
                    and_(
                        User.is_active == True,
                        Message.created_at <= cutoff,
                        text("(messages.tags->>'done')::boolean IS NOT TRUE"),
                        text("(messages.tags->>'follow_up_sent')::boolean IS NOT TRUE"),
                        text("messages.tags->'all_buckets' @> '\"To-Do\"'::jsonb"),
                        # Has at least one person entity
                        text(
                            "jsonb_array_length(messages.tags->'entities'->'people') > 0"
                        ),
                    )
                )
                .limit(20)
            )
            rows = result.all()
            for message, user in rows:
                try:
                    tags = message.tags if isinstance(message.tags, dict) else {}
                    tags["follow_up_sent"] = True
                    await session.execute(
                        update(Message)
                        .where(Message.id == message.id)
                        .values(tags=tags)
                    )
                except Exception as e:
                    print(f"[nudge] Follow-up mark failed for message {message.id}: {e}")
            await session.commit()


nudge_service = NudgeService()
