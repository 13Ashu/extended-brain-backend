import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env file directly - no export needed on Windows
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

from database import async_session_maker  
from models import Message
from services.embedding_service import embedding_service
from sqlalchemy import select, update

async def backfill():
    async with async_session_maker() as db:
        result = await db.execute(
            select(Message).where(Message.embedding == None)
        )
        messages = result.scalars().all()
        print(f"Backfilling {len(messages)} messages...")
        
        if not messages:
            print("No messages need backfilling!")
            return
        
        for i, msg in enumerate(messages):
            text = f"{msg.content} {msg.summary or ''}".strip()
            embedding = embedding_service.embed(text)
            await db.execute(
                update(Message).where(Message.id == msg.id).values(embedding=embedding)
            )
            if i % 50 == 0:
                await db.commit()
                print(f"  {i}/{len(messages)} done")
        
        await db.commit()
        print("✓ Backfill complete")

asyncio.run(backfill())