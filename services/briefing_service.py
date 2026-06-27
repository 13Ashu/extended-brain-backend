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
from sqlalchemy import and_, or_, select, update, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import async_session_maker, Message, User, Category, DeviceToken, GroupMember
from services.reminder_service import send_apns_notification
from services.group_service import total_unread_for_user


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
            # Find all active users whose briefing_time matches now (regardless of Telegram)
            users = await session.execute(
                select(User).where(
                    and_(
                        User.is_active == True,
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

            # ── Fetch tasks others assigned to me (shown in the iOS Today tab) ──
            assigned = await self._fetch_assigned_to_me(user.id, today, session)

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
        elif not assigned:
            text += "✨ *No tasks for today!*\n\n"

        if assigned:
            text += f"🤝 *Assigned to you ({len(assigned)})*\n"
            for t in assigned[:5]:
                text += f"• {t.content[:50]}\n"
            if len(assigned) > 5:
                text += f"_...and {len(assigned) - 5} more_\n"
            text += "\n"

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

        # ── APNs delivery (iOS primary surface) ───────────────────
        try:
            async with async_session_maker() as apns_db:
                result = await apns_db.execute(
                    select(DeviceToken).where(DeviceToken.user_id == user.id)
                )
                device_tokens = result.scalars().all()
                # Badge = total unread group messages (the app's single badge meaning),
                # not the task count — keeps the icon consistent across all push types.
                unread_badge = await total_unread_for_user(apns_db, user.id)

            personal_count = len(todos)
            assigned_count = len(assigned)
            event_count    = len(events)
            first_name = user.name.split()[0] if user.name else "there"

            # Clear, at-a-glance split so the user knows exactly what's pending and from
            # where: e.g. "4 personal · 3 assigned tasks today. Tap to see your plan."
            if personal_count and assigned_count:
                task_phrase = f"{personal_count} personal · {assigned_count} assigned tasks today"
            elif personal_count:
                task_phrase = f"{personal_count} personal task{'s' if personal_count != 1 else ''} today"
            elif assigned_count:
                task_phrase = f"{assigned_count} task{'s' if assigned_count != 1 else ''} assigned to you today"
            else:
                task_phrase = ""

            event_phrase = (
                f"{event_count} event{'s' if event_count != 1 else ''} today"
                if event_count else ""
            )

            if task_phrase and event_phrase:
                apns_body = f"{task_phrase} · {event_phrase}. Tap to see your plan."
            elif task_phrase:
                apns_body = f"{task_phrase}. Tap to see your plan."
            elif event_phrase:
                apns_body = f"{event_phrase}. Tap to see your schedule."
            else:
                apns_body = "Nothing due today. Have a great day!"

            for dt in device_tokens:
                await send_apns_notification(
                    device_token=dt.token,
                    title=f"Good morning, {first_name}! ☀️",
                    body=apns_body,
                    badge=unread_badge,
                    data={"type": "briefing"},
                    category="BRIEFING_TAP",
                )
        except Exception as e:
            print(f"[briefing] APNs delivery failed for user {user.id}: {e}")

        # ── Telegram delivery (legacy) ─────────────────────────────
        if not user.telegram_chat_id:
            return

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
                    Message.group_id.is_(None),          # exclude group-captured tasks
                    Message.assigned_to_user_id.is_(None),  # exclude mirror messages
                    text("messages.tags->>'due_date' < :today"),
                    text("(messages.tags->>'done')::boolean IS NOT TRUE"),
                    # Match the iOS To-Do tab exactly: a card counts as a task only when
                    # its PRIMARY bucket is To-Do, or it's a list. Using all_buckets @>
                    # 'To-Do' also swept in Events/Track items that merely had a secondary
                    # To-Do tag (e.g. "Dentist on Friday at 3pm") — iOS shows those as
                    # Events/Track, so the briefing over-counted and carry-forward also
                    # corrupted their real due dates by bumping them to today.
                    text("(messages.tags->>'primary_bucket' = 'To-Do' OR (messages.tags->>'is_list')::boolean IS TRUE)"),
                    text("(messages.tags->>'recurring')::boolean IS NOT TRUE"),
                )
            )
            .params(today=today)
        )
        overdue = result.scalars().all()

        for msg in overdue:
            tags = dict(msg.tags or {})
            tags["original_date"] = tags.get("due_date", "unknown")
            tags["due_date"]      = today
            tags["carried_over"]  = True
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
                    Message.group_id.is_(None),          # exclude group-captured tasks
                    Message.assigned_to_user_id.is_(None),  # exclude mirror messages
                    # Due today, or earlier and still open. Non-recurring overdue tasks
                    # were just bumped to today by _carry_forward, so the only `< today`
                    # rows left are the current occurrence of a recurring task (carry-
                    # forward leaves those alone). iOS shows that occurrence under Today
                    # too, so counting it here keeps the briefing == the Today tab.
                    text("messages.tags->>'due_date' <= :d"),
                    text("(messages.tags->>'done')::boolean IS NOT TRUE"),
                    # Same iOS-aligned bucket filter as _carry_forward (see note there):
                    # primary bucket To-Do, or a list — not a mere secondary To-Do tag.
                    text("(messages.tags->>'primary_bucket' = 'To-Do' OR (messages.tags->>'is_list')::boolean IS TRUE)"),
                )
            )
            .params(d=date_str)
            .order_by(Message.created_at.asc())
        )
        return result.scalars().all()

    async def _fetch_assigned_to_me(
        self, user_id: int, today: str, session: AsyncSession
    ) -> List[Message]:
        """Tasks others assigned to me that the iOS Today tab shows — so the briefing
        count matches the app. Same dual-case logic as GET /api/messages/assigned:
          Case 1 — explicit @mention (mirror row excluded via user_id != me), and MY
                   assignment slot is not already ticked off.
          Case 2 — group-wide unassigned To-Do where I'm a member.
        Non-future only (due is null or <= today): future-dated assignments live in the
        Upcoming tab, not Today. Done tasks excluded.
        """
        my_groups = select(GroupMember.group_id).where(GroupMember.user_id == user_id)
        result = await session.execute(
            select(Message)
            .where(
                and_(
                    or_(
                        and_(
                            Message.assigned_to_user_id == user_id,
                            Message.user_id != user_id,
                            text(
                                "NOT EXISTS (SELECT 1 FROM jsonb_array_elements("
                                "COALESCE(messages.tags->'assignments','[]'::jsonb)) a "
                                "WHERE (a->>'user_id')::int = :uid "
                                "AND COALESCE((a->>'done')::boolean,false) = true)"
                            ),
                        ),
                        and_(
                            Message.group_id.in_(my_groups),
                            Message.assigned_to_user_id.is_(None),
                            text("NOT (messages.tags ? 'assignments')"),
                            or_(
                                text("messages.tags->>'primary_bucket' = 'To-Do'"),
                                text("messages.tags->>'intent_bucket' = 'To-Do'"),
                            ),
                        ),
                    ),
                    text("COALESCE((messages.tags->>'done')::boolean, false) = false"),
                    text("(messages.tags->>'due_date' IS NULL OR messages.tags->>'due_date' <= :today)"),
                )
            )
            .params(uid=user_id, today=today)
        )
        # Dedup by id (the OR can surface a row via either branch).
        seen: set[int] = set()
        out: List[Message] = []
        for m in result.scalars().all():
            if m.id not in seen:
                seen.add(m.id)
                out.append(m)
        return out

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
