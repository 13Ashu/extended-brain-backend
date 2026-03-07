"""
Migration: Re-classify all existing messages into intent buckets.

Run once:
    python migrate_intent_buckets.py

What it does:
  1. Loads every message from the DB
  2. Calls the LLM to assign an intent bucket (Remember / To-Do / Ideas / Track / Events / Random)
  3. Ensures the 6 bucket categories exist for each user
  4. Updates message.category_id and message.tags["intent_bucket"]
  5. Prints a summary at the end

Safe to re-run — already-correct messages are skipped if you pass --skip-correct.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update
from datetime import datetime

# ── adjust these imports to match your project layout ──
from database import User, Message, Category
from cerebras_client import CerebrasClient


# ─────────────────────────────────────────────────────────────────────────────
# Intent bucket definitions (must match message_processor.py)
# ─────────────────────────────────────────────────────────────────────────────

INTENT_BUCKETS = {
    "Remember": (
        "User wants to remember a fact, location, or piece of information for later. "
        "Examples: 'parked at C3', 'nailcutter in 3rd drawer', 'wifi password is 1234'."
    ),
    "To-Do": (
        "User has a task, action item, or reminder for themselves. "
        "Examples: 'buy milk', 'call dentist', 'submit report by Friday'."
    ),
    "Ideas": (
        "User is capturing a new idea, thought, concept, or creative spark. "
        "Examples: 'what if we built X', 'startup idea: Y', 'feature idea for the app'."
    ),
    "Track": (
        "User wants to log or monitor something over time — health, habits, progress, mood. "
        "Examples: 'weight today 74kg', 'ran 5km', 'mood: anxious today'."
    ),
    "Events": (
        "User is noting a time-anchored event, appointment, or plan. "
        "Examples: 'mom visiting 10th march', 'dentist appointment Friday 3pm'."
    ),
    "Random": (
        "Casual, venting, conversational, or unclear intent. "
        "Examples: 'heyy', 'lol okay', 'testing 123', single word messages."
    ),
}

BUCKET_NAMES = list(INTENT_BUCKETS.keys())

# ─────────────────────────────────────────────────────────────────────────────
# LLM classification (same prompt as message_processor.py)
# ─────────────────────────────────────────────────────────────────────────────

async def classify_message(content: str, cerebras: CerebrasClient) -> str:
    bucket_descriptions = "\n".join(
        f'  "{name}": {desc}' for name, desc in INTENT_BUCKETS.items()
    )

    prompt = f"""You are classifying a personal note by WHY the user saved it — their INTENT.

AVAILABLE BUCKETS (these are the ONLY valid options):
{bucket_descriptions}

NOTE: "{content}"

RULES:
1. Pick exactly ONE bucket.
2. Ask: WHY did the user write this?
   - To recall it later?        → Remember
   - To act on it?              → To-Do
   - To explore an idea?        → Ideas
   - To log progress/habit?     → Track
   - To note a future event?    → Events
   - Just venting / casual?     → Random
3. NEVER return a topic name like "Parking", "Cosmetics", "Food".

Return ONLY this JSON:
{{"bucket": "one of the 6 bucket names above"}}"""

    response = await cerebras.chat(prompt)
    bucket = response.get("bucket", "Random")
    return bucket if bucket in BUCKET_NAMES else "Random"


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

async def get_or_create_bucket_category(
    user_id: int,
    bucket_name: str,
    db: AsyncSession,
) -> Category:
    result = await db.execute(
        select(Category).where(
            Category.user_id == user_id,
            Category.name == bucket_name,
        )
    )
    cat = result.scalar_one_or_none()
    if not cat:
        cat = Category(
            user_id=user_id,
            name=bucket_name,
            description=INTENT_BUCKETS[bucket_name],
        )
        db.add(cat)
        await db.flush()
    return cat


# ─────────────────────────────────────────────────────────────────────────────
# Main migration
# ─────────────────────────────────────────────────────────────────────────────

async def migrate():
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set in .env")
        sys.exit(1)

    # SQLAlchemy async engine
    async_url = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine = create_async_engine(async_url, echo=False)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    cerebras = CerebrasClient()

    skip_correct = "--skip-correct" in sys.argv

    async with AsyncSessionLocal() as db:
        # Load all messages with their current category
        result = await db.execute(
            select(Message, Category)
            .join(Category, Message.category_id == Category.id, isouter=True)
            .order_by(Message.created_at.asc())
        )
        rows = result.all()

        total   = len(rows)
        skipped = 0
        updated = 0
        failed  = 0

        print(f"\n🔄 Starting migration of {total} messages...\n")

        # Pre-create all 6 bucket categories for every user
        user_ids = {m.user_id for m, _ in rows}
        print(f"👥 Found {len(user_ids)} user(s). Pre-creating intent buckets...")
        for uid in user_ids:
            for bucket_name in BUCKET_NAMES:
                await get_or_create_bucket_category(uid, bucket_name, db)
        await db.flush()
        print("✓ Buckets ready\n")

        for i, (message, old_category) in enumerate(rows, 1):
            old_cat_name = old_category.name if old_category else "None"

            # Skip if already a valid intent bucket
            if skip_correct and old_cat_name in BUCKET_NAMES:
                current_bucket = message.tags.get("intent_bucket") if isinstance(message.tags, dict) else None
                if current_bucket in BUCKET_NAMES:
                    skipped += 1
                    print(f"  [{i}/{total}] ⏭  Already classified: '{old_cat_name}' — {message.content[:60]}")
                    continue

            try:
                # Classify
                bucket = await classify_message(message.content, cerebras)

                # Get the bucket category for this user
                bucket_cat = await get_or_create_bucket_category(message.user_id, bucket, db)

                # Update tags
                tags = message.tags if isinstance(message.tags, dict) else {}
                tags["intent_bucket"] = bucket

                # Update message
                await db.execute(
                    update(Message)
                    .where(Message.id == message.id)
                    .values(category_id=bucket_cat.id, tags=tags)
                )

                updated += 1
                arrow = "→" if old_cat_name != bucket else "="
                print(f"  [{i}/{total}] ✓  '{old_cat_name}' {arrow} {bucket} | {message.content[:60]}")

            except Exception as e:
                failed += 1
                print(f"  [{i}/{total}] ✗  Failed: {e} | {message.content[:60]}")

            # Commit in batches of 20
            if i % 20 == 0:
                await db.commit()
                print(f"\n  💾 Committed batch at {i}/{total}\n")

        # Final commit
        await db.commit()

        print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Migration complete!
   Total   : {total}
   Updated : {updated}
   Skipped : {skipped}
   Failed  : {failed}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(migrate())


# ─────────────────────────────────────────────────────────────────────────────
# Due-date backfill (run after main migration)
# ─────────────────────────────────────────────────────────────────────────────

async def backfill_due_dates():
    """
    Second pass: for all messages already in the To-Do bucket that are
    missing tags.due_date, infer and write the due date.
    """
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    async_url    = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    engine       = create_async_engine(async_url, echo=False)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    cerebras = CerebrasClient()

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Message)
            .join(Category, Message.category_id == Category.id)
            .where(Category.name == "To-Do")
        )
        todos = result.scalars().all()

        print(f"\n📅 Backfilling due dates for {len(todos)} To-Do messages...\n")
        updated = 0

        for i, message in enumerate(todos, 1):
            tags = message.tags if isinstance(message.tags, dict) else {}
            if tags.get("due_date"):
                print(f"  [{i}/{len(todos)}] ⏭  Already has due_date: {message.content[:60]}")
                continue

            try:
                from datetime import timedelta
                ref = message.created_at or datetime.utcnow()

                prompt = f"""Extract or infer a due date for this to-do.
Today (when it was saved): {ref.strftime('%Y-%m-%d')} ({ref.strftime('%A')})
To-do: "{message.content}"
Rules:
- Explicit date → resolve to YYYY-MM-DD
- No date mentioned → always use today: {ref.strftime('%Y-%m-%d')}
- NEVER return null
Return ONLY: {{"due_date": "YYYY-MM-DD"}}"""

                response = await cerebras.chat(prompt, max_tokens=80)
                due = response.get("due_date")
                import re
                if due and re.match(r"^\d{4}-\d{2}-\d{2}$", str(due)):
                    tags["due_date"] = due
                else:
                    tags["due_date"] = None

                await db.execute(
                    update(Message).where(Message.id == message.id).values(tags=tags)
                )
                updated += 1
                print(f"  [{i}/{len(todos)}] ✓  due={tags['due_date']} | {message.content[:60]}")

            except Exception as e:
                print(f"  [{i}/{len(todos)}] ✗  {e} | {message.content[:60]}")

            if i % 20 == 0:
                await db.commit()

        await db.commit()
        print(f"\n✅ Due date backfill complete — {updated} updated\n")

    await engine.dispose()


if __name__ == "__main__":
    import sys
    if "--due-dates" in sys.argv:
        asyncio.run(backfill_due_dates())
    else:
        asyncio.run(migrate())
