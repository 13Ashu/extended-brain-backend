"""
Nudge Service
─────────────────────────────────────────────────────────────
• Idle nudges  — todos not acted on since yesterday get daily pings
                 with Snooze / Done inline buttons
• Follow-up tracking — messages with people + action verbs get a
                       follow-up ping after 48hrs if not marked done
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import and_, select, update, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import async_session_maker, Message, User

IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

FOLLOWUP_HOURS = 48


async def _send_telegram(chat_id: str, text_: str, reply_markup: dict):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    async with httpx.AsyncClient(timeout=15.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id":      chat_id,
                "text":         text_,
                "parse_mode":   "Markdown",
                "reply_markup": reply_markup,
            },
        )


class NudgeService:

    # ──────────────────────────────────────────────────────────────
    # Idle nudges
    # ──────────────────────────────────────────────────────────────

    async def run_idle_nudges(self):
        """
        Called once per day (at briefing time + 1 min from scheduler).
        Finds todos created/due before today, not done, not snoozed until tomorrow.
        """
        now_utc = datetime.utcnow()
        today   = datetime.now(IST).strftime("%Y-%m-%d")

        async with async_session_maker() as session:
            users = await session.execute(
                select(User).where(
                    and_(
                        User.is_active == True,
                        User.telegram_chat_id.isnot(None),
                    )
                )
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

    async def _send_nudge(self, chat_id: str, message: Message):
        tags     = message.tags if isinstance(message.tags, dict) else {}
        due      = tags.get("due_date", "")
        due_str  = f" _(due {due})_" if due else ""
        priority = tags.get("priority", "normal")
        prefix   = "🔴" if priority in ("high", "urgent") else "📌"

        text_ = (
            f"{prefix} *Pending task reminder*\n\n"
            f"{message.content[:100]}{due_str}\n\n"
            f"_What do you want to do with this?_"
        )
        reply_markup = {
            "inline_keyboard": [[
                {"text": "✅ Done",        "callback_data": f"done:{message.id}"},
                {"text": "⏰ Snooze 1 day", "callback_data": f"snooze:{message.id}:1440"},
            ]]
        }
        await _send_telegram(chat_id, text_, reply_markup)

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
        """
        cutoff = datetime.utcnow() - timedelta(hours=FOLLOWUP_HOURS)

        async with async_session_maker() as session:
            result = await session.execute(
                select(Message, User)
                .join(User, Message.user_id == User.id)
                .where(
                    and_(
                        User.is_active == True,
                        User.telegram_chat_id.isnot(None),
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
                await self._send_followup(user, message)
            except Exception as e:
                print(f"[nudge] Follow-up failed for message {message.id}: {e}")

    async def _send_followup(self, user: User, message: Message):
        tags    = message.tags if isinstance(message.tags, dict) else {}
        people  = tags.get("entities", {}).get("people", [])
        actions = tags.get("actionables", [])

        if not people:
            return

        person      = people[0]
        action_hint = actions[0] if actions else message.content[:50]

        text_ = (
            f"🔔 *Follow-up check*\n\n"
            f"Did you connect with *{person}*?\n"
            f"_{action_hint}_\n\n"
            f"_Saved {message.created_at.strftime('%d %b at %I:%M %p')}_"
        )
        reply_markup = {
            "inline_keyboard": [[
                {"text": "✅ Yes, done",      "callback_data": f"done:{message.id}"},
                {"text": "⏰ Remind tomorrow", "callback_data": f"snooze:{message.id}:1440"},
            ]]
        }

        await _send_telegram(user.telegram_chat_id, text_, reply_markup)

        # Mark follow_up_sent so we don't spam
        async with async_session_maker() as session:
            tags["follow_up_sent"] = True
            await session.execute(
                update(Message)
                .where(Message.id == message.id)
                .values(tags=tags)
            )
            await session.commit()


nudge_service = NudgeService()
