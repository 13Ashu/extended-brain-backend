"""
Database Migration Script
Run this to update your database schema with new user fields
"""

import asyncio
import sys
from sqlalchemy import text
from dotenv import load_dotenv
import os

# Load environment variables FIRST
load_dotenv()

# Verify DATABASE_URL is loaded
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("❌ DATABASE_URL not found in .env file!")
    sys.exit(1)

print(f"✓ Using database: {DATABASE_URL.split('@')[1].split('/')[0]}")

from database import engine, async_session_maker


async def check_column_exists(table: str, column: str) -> bool:
    """Check if a column exists in a table"""
    async with async_session_maker() as session:
        query = text(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='{table}' AND column_name='{column}'
        """)
        print(query)
        result = await session.execute(query)
        return result.fetchone() is not None


async def add_column_if_not_exists(table: str, column: str, definition: str):
    """Add a column if it doesn't exist"""
    exists = await check_column_exists(table, column)
    
    if not exists:
        async with async_session_maker() as session:
            query = text(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            await session.execute(query)
            await session.commit()
            print(f"✓ Added column {column} to {table}")
    else:
        print(f"⊘ Column {column} already exists in {table}")


async def migrate_user_table():
    """Add new user registration fields"""
    print("\n🔄 Migrating users table...")
    
    # Add email column
    await add_column_if_not_exists(
        'users', 
        'email', 
        'VARCHAR(255) UNIQUE NOT NULL DEFAULT \'\''
    )
    
    # Add age column
    await add_column_if_not_exists(
        'users', 
        'age', 
        'INTEGER DEFAULT 0'
    )
    
    # Add occupation column
    await add_column_if_not_exists(
        'users', 
        'occupation', 
        'VARCHAR(100) DEFAULT \'\''
    )
    
    # Add password_hash column
    await add_column_if_not_exists(
        'users', 
        'password_hash', 
        'VARCHAR(255) DEFAULT \'\''
    )
    
    # Add is_active column
    await add_column_if_not_exists(
        'users', 
        'is_active', 
        'BOOLEAN DEFAULT TRUE'
    )
    
    # Add is_verified column
    await add_column_if_not_exists(
        'users', 
        'is_verified', 
        'BOOLEAN DEFAULT FALSE'
    )
    
    # Add last_login column
    await add_column_if_not_exists(
        'users', 
        'last_login', 
        'TIMESTAMP'
    )
    
    print("✓ Users table migration complete\n")


async def create_otp_table():
    """Create OTP verification table if it doesn't exist"""
    print("🔄 Creating otp_verifications table...")
    
    async with async_session_maker() as session:
        create_table_query = text("""
            CREATE TABLE IF NOT EXISTS otp_verifications (
                id SERIAL PRIMARY KEY,
                phone_number VARCHAR(20) NOT NULL,
                otp_code VARCHAR(6) NOT NULL,
                is_verified BOOLEAN DEFAULT FALSE,
                attempts INTEGER DEFAULT 0,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        await session.execute(create_table_query)
        
        # Create index on phone_number
        create_index_query = text("""
            CREATE INDEX IF NOT EXISTS idx_otp_phone 
            ON otp_verifications(phone_number)
        """)
        
        await session.execute(create_index_query)
        await session.commit()
        
    print("✓ OTP verifications table created\n")


async def create_indexes():
    """Create necessary indexes for performance"""
    print("🔄 Creating indexes...")
    
    async with async_session_maker() as session:
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_phone ON users(phone_number)",
            "CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_category ON messages(category_id)",
            "CREATE INDEX IF NOT EXISTS idx_categories_user ON categories(user_id)",
        ]
        
        for index_query in indexes:
            await session.execute(text(index_query))
        
        await session.commit()
    
    print("✓ Indexes created\n")


async def verify_migration():
    """Verify the migration was successful"""
    print("🔍 Verifying migration...")
    
    async with async_session_maker() as session:
        # Check users table structure
        query = text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name='users'
            ORDER BY ordinal_position
        """)
        
        result = await session.execute(query)
        columns = result.fetchall()
        
        print("\n📊 Users table structure:")
        for col in columns:
            print(f"  - {col[0]}: {col[1]}")
        
        # Check OTP table
        query = text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name='otp_verifications'
            ORDER BY ordinal_position
        """)
        
        result = await session.execute(query)
        columns = result.fetchall()
        
        print("\n📊 OTP Verifications table structure:")
        for col in columns:
            print(f"  - {col[0]}: {col[1]}")
    
    print("\n✅ Migration verification complete\n")


async def main():
    """Run all migrations"""
    print("=" * 60)
    print("🚀 Extended Brain Database Migration")
    print("=" * 60)
    
    try:
        # Run migrations
        await migrate_user_table()
        await create_otp_table()
        await create_indexes()
        await verify_migration()
        
        print("=" * 60)
        print("✅ All migrations completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
    
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
