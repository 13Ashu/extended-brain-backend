"""
Database Migration: Upgrade embedding column to 3072 dimensions
────────────────────────────────────────────────────────────────────────────
Run this ONCE before deploying the new code.

Usage:
  python migrate_embeddings.py

This will:
  1. Drop the existing embedding column (was vector(384))
  2. Re-create it as vector(3072)
  3. Create an HNSW index for fast cosine similarity search
  4. Optionally re-embed all existing messages (recommended)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "").replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

NEW_DIMS = 1536


async def migrate():
    async with engine.begin() as conn:
        print("Step 1: Ensuring pgvector extension is enabled...")
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        print("Step 2: Dropping old embedding column...")
        await conn.execute(text("ALTER TABLE messages DROP COLUMN IF EXISTS embedding"))

        print(f"Step 3: Adding new embedding column (vector({NEW_DIMS}))...")
        await conn.execute(text(f"ALTER TABLE messages ADD COLUMN embedding vector({NEW_DIMS})"))

        print("Step 4: Creating HNSW index for fast cosine search...")
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS messages_embedding_hnsw "
            "ON messages USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        ))

        print("✅ Schema migration complete!")

    # Optional: re-embed all existing messages
    answer = input("\nRe-embed all existing messages with new 3072-dim model? (y/N): ").strip().lower()
    if answer != "y":
        print("Skipping re-embedding. Existing messages won't have embeddings until updated.")
        return

    print("\nRe-embedding all messages...")
    from services.embedding_service import embedding_service
    from database import Message

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            text("SELECT id, content, summary, tags FROM messages ORDER BY id")
        )
        rows = result.fetchall()
        print(f"Found {len(rows)} messages to re-embed.")

        from sqlalchemy import update
        from database import Message as MsgModel

        for i, row in enumerate(rows):
            msg_id  = row.id
            content = row.content or ""
            summary = row.summary or ""
            tags    = row.tags or {}

            keywords   = tags.get("keywords", []) if isinstance(tags, dict) else []
            actionables = tags.get("actionables", []) if isinstance(tags, dict) else []
            buckets    = tags.get("all_buckets", []) if isinstance(tags, dict) else []

            enriched = " ".join(filter(None, [
                content, summary,
                " ".join(keywords),
                " ".join(actionables),
                " ".join(buckets),
            ]))

            try:
                embedding = embedding_service.embed(enriched.strip())
                await db.execute(
                    update(MsgModel)
                    .where(MsgModel.id == msg_id)
                    .values(embedding=embedding)
                )
                if (i + 1) % 10 == 0:
                    await db.commit()
                    print(f"  ✓ Re-embedded {i + 1}/{len(rows)} messages")
            except Exception as e:
                print(f"  ⚠ Failed to embed message {msg_id}: {e}")

        await db.commit()
        print(f"✅ Re-embedding complete! {len(rows)} messages updated.")


if __name__ == "__main__":
    asyncio.run(migrate())
