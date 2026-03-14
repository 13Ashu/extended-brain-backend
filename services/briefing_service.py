"""
Briefing Service
─────────────────────────────────────────────────────────────
• Morning briefing — daily summary of todos + events
• Carry-forward — overdue todos get bumped to today
• Runs inside the existing reminder scheduler loop
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import and_, select, update, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import async_session_maker, Message, User, Category


IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

BRIEFING_TIME_KEY = "briefing_time"  # stored as "HH:MM" in user row


async def _send_telegram(chat_id: str, text: str, reply_markup: Optional[dict] = None):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    payload = {
        "chat_id":    chat_id,
        "text":       text,
        "parse_mode": "Markdown",
    }
    if reply_markup:
        payload["reply_markup"] = reply_markup

    async with httpx.AsyncClient(timeout=15.0) as client:
        await client.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=payload,
        )


class BriefingService:

    async def run(self):
        """
        Called every minute from the scheduler.
        Checks if any user's briefing time matches current IST minute.
        """
        now_ist = datetime.now(IST)
        current_hhmm = now_ist.strftime("%H:%M")

        async with async_session_maker() as session:
            # Find all active users whose briefing_time matches now
            users = await session.execute(
                select(User).where(
                    and_(
                        User.is_active == True,
                        User.telegram_chat_id.isnot(None),
                        User.briefing_time == current_hhmm,
                    )
                )
            )
            users = users.scalars().all()

        for user in users:
            try:
                await self._send_briefing(user)
            except Exception as e:
                print(f"[briefing] Failed for user {user.id}: {e}")

    async def _send_briefing(self, user: User):
        """Build and send the morning briefing for one user."""
        now_ist  = datetime.now(IST)
        today    = now_ist.strftime("%Y-%m-%d")
        tomorrow = (now_ist + timedelta(days=1)).strftime("%Y-%m-%d")

        async with async_session_maker() as session:
            # ── Carry forward overdue todos ───────────────────────
            carried = await self._carry_forward(user.id, today, session)

            # ── Fetch today's todos ───────────────────────────────
            todos = await self._fetch_todos(user.id, today, session)

            # ── Fetch today's events ──────────────────────────────
            events = await self._fetch_events(user.id, today, session)

            # ── Fetch tomorrow's events (preview) ────────────────
            tomorrow_events = await self._fetch_events(user.id, tomorrow, session)

        # ── Build message ─────────────────────────────────────────
        greeting = self._greeting(now_ist.hour, user.name)
        text = f"{greeting}\n\n"

        if carried:
            text += f"⏪ *{len(carried)} task(s) carried over from yesterday*\n\n"

        if todos:
            text += f"📋 *Today's Tasks ({len(todos)})*\n"
            for t in todos[:8]:
                tags     = t.tags if isinstance(t.tags, dict) else {}
                priority = tags.get("priority", "normal")
                prefix   = "🔴" if priority == "high" else "🟡" if priority == "urgent" else "•"
                evt_time = tags.get("event_time", "")
                time_str = f" _{evt_time}_" if evt_time else ""
                text += f"{prefix} {t.content[:50]}{time_str}\n"
            if len(todos) > 8:
                text += f"_...and {len(todos) - 8} more_\n"
            text += "\n"
        else:
            text += "✨ *No tasks for today!*\n\n"

        if events:
            text += f"📅 *Today's Events ({len(events)})*\n"
            for e in events[:5]:
                tags     = e.tags if isinstance(e.tags, dict) else {}
                evt_time = tags.get("event_time", "")
                time_str = f" at _{evt_time}_" if evt_time else ""
                text += f"• {e.content[:50]}{time_str}\n"
            text += "\n"

        if tomorrow_events:
            text += f"👀 *Tomorrow: {len(tomorrow_events)} event(s)*\n"
            for e in tomorrow_events[:3]:
                text += f"• {e.content[:40]}\n"
            text += "\n"

        text += "_Reply 'search: your query' to find anything_"

        # Inline button to change briefing time
        reply_markup = {
            "inline_keyboard": [[
                {"text": "⏰ Change briefing time", "callback_data": "set_briefing_time"}
            ]]
        }

        await _send_telegram(user.telegram_chat_id, text, reply_markup)
        print(f"[briefing] Sent to {user.name} ({user.telegram_chat_id})")

    async def _carry_forward(
        self, user_id: int, today: str, session: AsyncSession
    ) -> List[Message]:
        """
        Find todos with due_date < today and done != true.
        Update their due_date to today and tag them carried_over.
        """
        result = await session.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    text("messages.tags->>'due_date' < :today"),
                    text("(messages.tags->>'done')::boolean IS NOT TRUE"),
                    text("messages.tags->'all_buckets' @> '\"To-Do\"'::jsonb"),
                )
            )
            .params(today=today)
        )
        overdue = result.scalars().all()

        for msg in overdue:
            tags = dict(msg.tags or {})
            tags["due_date"]      = today
            tags["carried_over"]  = True
            tags["original_date"] = tags.get("due_date", "unknown")
            await session.execute(
                update(Message)
                .where(Message.id == msg.id)
                .values(tags=tags)
            )

        if overdue:
            await session.commit()

        return overdue

    async def _fetch_todos(
        self, user_id: int, date_str: str, session: AsyncSession
    ) -> List[Message]:
        result = await session.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    text("messages.tags->>'due_date' = :d"),
                    text("(messages.tags->>'done')::boolean IS NOT TRUE"),
                    text("messages.tags->'all_buckets' @> '\"To-Do\"'::jsonb"),
                )
            )
            .params(d=date_str)
            .order_by(Message.created_at.asc())
        )
        return result.scalars().all()

    async def _fetch_events(
        self, user_id: int, date_str: str, session: AsyncSession
    ) -> List[Message]:
        result = await session.execute(
            select(Message)
            .where(
                and_(
                    Message.user_id == user_id,
                    text(
                        "EXISTS ("
                        "  SELECT 1 FROM jsonb_array_elements(messages.tags->'events') ev"
                        "  WHERE ev->>'date' = :d"
                        ")"
                    ),
                )
            )
            .params(d=date_str)
            .order_by(Message.created_at.asc())
        )
        return result.scalars().all()

    def _greeting(self, hour: int, name: str) -> str:
        first_name = name.split()[0]
        if hour < 12:
            return f"🌅 Good morning, {first_name}!"
        elif hour < 17:
            return f"☀️ Good afternoon, {first_name}!"
        else:
            return f"🌙 Good evening, {first_name}!"

    async def set_briefing_time(self, user_id: int, hhmm: str) -> bool:
        """Update user's briefing time. hhmm format: 'HH:MM'"""
        try:
            async with async_session_maker() as session:
                await session.execute(
                    update(User)
                    .where(User.id == user_id)
                    .values(briefing_time=hhmm)
                )
                await session.commit()
            return True
        except Exception as e:
            print(f"[briefing] Failed to set briefing time: {e}")
            return False


briefing_service = BriefingService()
