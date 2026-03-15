"""
Database models using SQLAlchemy with PostgreSQL
Updated with full user registration fields
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Text, DateTime, ForeignKey, Enum as SQLEnum, Integer
from sqlalchemy.dialects.postgresql import JSON, JSONB
from datetime import datetime
from typing import Optional, List, AsyncGenerator
from pgvector.sqlalchemy import Vector
import enum
import os


# Database URL from environment (Neon PostgreSQL)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://user:password@localhost/extendedbrain"
)

# Clean up the URL - remove sslmode and convert to asyncpg
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

# Normalize URL for asyncpg
if DATABASE_URL:
    # Strip any query params (ssl handled via connect_args instead)
    if "?" in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.split("?")[0]
    
    # Ensure asyncpg driver prefix
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif DATABASE_URL.startswith("postgres://"):   # Neon sometimes emits this form
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

# Create async engine
import ssl as _ssl

# Build an SSL context asyncpg actually understands
_ssl_ctx = _ssl.create_default_context()
_ssl_ctx.check_hostname = False          # Neon's hostname is fine, but this avoids cert issues
_ssl_ctx.verify_mode   = _ssl.CERT_NONE # Neon uses self-signed intermediates on some regions

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=5,          # Neon free tier has a connection limit — don't go above 10
    max_overflow=2,
    connect_args={
        "ssl":     _ssl_ctx,
        "timeout": 30,    # asyncpg connection timeout in seconds
        "server_settings": {
            "application_name": "extended_brain",
        },
    },
)

# Create session maker
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


# Base class for models
class Base(DeclarativeBase):
    pass


# Enums
class MessageType(enum.Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    DOCUMENT = "document"
    VIDEO = "video"


# Models
class User(Base):
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    telegram_chat_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, unique=True)
    
    # Contact Information
    phone_number: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    
    # Personal Information
    name: Mapped[str] = mapped_column(String(100))
    age: Mapped[int] = mapped_column(Integer)
    occupation: Mapped[str] = mapped_column(String(100))
    
    # Authentication
    password_hash: Mapped[str] = mapped_column(String(255))  # Store hashed password
    
    # Settings
    timezone: Mapped[str] = mapped_column(String(50), default="Asia/Kolkata")
    briefing_time: Mapped[Optional[str]] = mapped_column(String(5), default="08:00", nullable=True)  # ← ADD THIS
    is_active: Mapped[bool] = mapped_column(default=True)
    is_verified: Mapped[bool] = mapped_column(default=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    categories: Mapped[List["Category"]] = relationship(
        "Category",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<User {self.phone_number} - {self.name}>"


class OTPVerification(Base):
    """Store OTP codes for phone verification"""
    __tablename__ = "otp_verifications"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    phone_number: Mapped[str] = mapped_column(String(20), index=True)
    otp_code: Mapped[str] = mapped_column(String(6))
    is_verified: Mapped[bool] = mapped_column(default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<OTP {self.phone_number} - {self.otp_code}>"


class Category(Base):
    __tablename__ = "categories"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    description: Mapped[Optional[str]] = mapped_column(Text)
    color: Mapped[Optional[str]] = mapped_column(String(7))  # Hex color
    icon: Mapped[Optional[str]] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="categories")
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="category"
    )
    
    def __repr__(self):
        return f"<Category {self.name}>"


class Message(Base):
    __tablename__ = "messages"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    category_id: Mapped[Optional[int]] = mapped_column(ForeignKey("categories.id"), index=True)
    
    content: Mapped[str] = mapped_column(Text)
    message_type: Mapped[MessageType] = mapped_column(SQLEnum(MessageType))
    media_url: Mapped[Optional[str]] = mapped_column(String(500))
    
    # AI-generated metadata
    summary: Mapped[Optional[str]] = mapped_column(Text)
    tags: Mapped[Optional[dict]] = mapped_column(JSONB)  # List of tags
    entities: Mapped[Optional[dict]] = mapped_column(JSONB)  # Extracted entities
    embedding: Mapped[Optional[List[float]]] = mapped_column(Vector(1536), nullable=True)  # Vector embedding for search
    
    # Timestamps
    original_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="messages")
    category: Mapped[Optional["Category"]] = relationship("Category", back_populates="messages")
    
    def __repr__(self):
        return f"<Message {self.id} - {self.message_type.value}>"


# Database initialization
async def init_db():
    """Create all tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
