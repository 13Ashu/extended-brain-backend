# test_db.py
import asyncio
from dotenv import load_dotenv
import os
from sqlalchemy import text

load_dotenv()

async def test():
    from models import engine
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT version()"))
        print("✓ Connected to:", result.scalar())

asyncio.run(test())