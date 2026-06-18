"""
Database models using SQLAlchemy with PostgreSQL (Railway)
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Text, DateTime, ForeignKey, Enum as SQLEnum, Integer, Boolean, LargeBinary, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSON, JSONB
from datetime import datetime
from typing import Optional, List, AsyncGenerator
from pgvector.sqlalchemy import Vector
import enum
import os


# Database URL from environment (Railway PostgreSQL)
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

# Normalize URL for asyncpg
if "?" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split("?")[0]

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=5,
    connect_args={
        "ssl": False,     # Railway internal networking — no SSL needed
        "timeout": 30,
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
    phone_number: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    
    # Personal Information
    name: Mapped[str] = mapped_column(String(100))
    age: Mapped[int] = mapped_column(Integer)
    occupation: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, default="")
    
    # Authentication
    password_hash: Mapped[str] = mapped_column(String(255))  # Store hashed password
    google_uid: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, unique=True, index=True)
    apple_uid: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, unique=True, index=True)
    
    # Settings
    timezone: Mapped[str] = mapped_column(String(50), default="Asia/Kolkata")
    briefing_time: Mapped[Optional[str]] = mapped_column(String(5), default="08:00", nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    is_verified: Mapped[bool] = mapped_column(default=False)
    is_pro: Mapped[bool] = mapped_column(Boolean, default=False)
    active_group_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
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
        cascade="all, delete-orphan",
        foreign_keys="Message.user_id",
    )
    categories: Mapped[List["Category"]] = relationship(
        "Category",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    pro_account: Mapped[Optional["ProAccount"]] = relationship(
        "ProAccount", back_populates="owner", uselist=False,
        foreign_keys="ProAccount.owner_id",
    )
    group_memberships: Mapped[List["GroupMember"]] = relationship(
        "GroupMember", back_populates="user", cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<User {self.phone_number} - {self.name}>"


class DeviceToken(Base):
    """APNs device tokens for push notifications"""
    __tablename__ = "device_tokens"

    id:         Mapped[int]      = mapped_column(primary_key=True)
    user_id:    Mapped[int]      = mapped_column(ForeignKey("users.id"), index=True)
    token:      Mapped[str]      = mapped_column(String(255), unique=True, index=True)
    platform:   Mapped[str]      = mapped_column(String(20), default="ios")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<DeviceToken user={self.user_id} token={self.token[:8]}...>"


class OTPVerification(Base):
    """Store OTP codes for phone verification"""
    __tablename__ = "otp_verifications"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    phone_number: Mapped[str] = mapped_column(String(20), index=True)
    otp_code: Mapped[str] = mapped_column(String(20))
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
    group_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    assigned_to_user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

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
    user: Mapped["User"] = relationship("User", back_populates="messages", foreign_keys=[user_id])
    category: Mapped[Optional["Category"]] = relationship("Category", back_populates="messages")
    
    def __repr__(self):
        return f"<Message {self.id} - {self.message_type.value}>"


class ProAccount(Base):
    """One Pro account per paying user. Holds up to max_members members."""
    __tablename__ = "pro_accounts"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), unique=True, index=True)
    plan_type: Mapped[str] = mapped_column(String(20), default="pro")
    max_members: Mapped[int] = mapped_column(Integer, default=10)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    owner: Mapped["User"] = relationship("User", back_populates="pro_account", foreign_keys=[owner_id])
    members: Mapped[List["ProAccountMember"]] = relationship(
        "ProAccountMember", back_populates="account", cascade="all, delete-orphan"
    )
    groups: Mapped[List["Group"]] = relationship(
        "Group", back_populates="account", cascade="all, delete-orphan"
    )


class ProAccountMember(Base):
    """Invited members of a Pro account. Pending until they accept."""
    __tablename__ = "pro_account_members"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("pro_accounts.id"), index=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    phone_number: Mapped[str] = mapped_column(String(100), index=True)
    invited_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    invite_token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    invited_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    joined_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    account: Mapped["ProAccount"] = relationship("ProAccount", back_populates="members")


class Group(Base):
    """Collaborative group within a Pro account.

    Membership is now self-contained per group (WhatsApp-style): anyone with the
    `invite_token` link can join, up to `max_members`. There is no account-wide
    roster — the parent ProAccount only anchors ownership/billing.
    """
    __tablename__ = "groups"

    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("pro_accounts.id"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    emoji: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    photo_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    # Shareable join link token (minted at creation). Unique so /join/{token} resolves.
    invite_token: Mapped[Optional[str]] = mapped_column(String(64), unique=True, index=True, nullable=True)
    # Per-group member cap (includes the creator/admin).
    max_members: Mapped[int] = mapped_column(Integer, default=10)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    account: Mapped["ProAccount"] = relationship("ProAccount", back_populates="groups")
    members: Mapped[List["GroupMember"]] = relationship(
        "GroupMember", back_populates="group", cascade="all, delete-orphan"
    )


class GroupMember(Base):
    """Membership in a group."""
    __tablename__ = "group_members"

    id: Mapped[int] = mapped_column(primary_key=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(20), default="member")
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    group: Mapped["Group"] = relationship("Group", back_populates="members")
    user: Mapped["User"] = relationship("User", back_populates="group_memberships")


class CouponCode(Base):
    """Coupon codes for Pro plan — free trials or discounts."""
    __tablename__ = "coupon_codes"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    # 'free' = full Pro access, 'percent' = % off, 'fixed' = fixed amount off
    discount_type: Mapped[str] = mapped_column(String(20), default="free")
    discount_value: Mapped[int] = mapped_column(Integer, default=100)  # % or ₹ amount
    duration_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # None = forever
    max_uses: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # None = unlimited
    uses_count: Mapped[int] = mapped_column(Integer, default=0)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    redemptions: Mapped[List["CouponRedemption"]] = relationship(
        "CouponRedemption", back_populates="coupon", cascade="all, delete-orphan"
    )


class CouponRedemption(Base):
    """Tracks who redeemed which coupon."""
    __tablename__ = "coupon_redemptions"

    id: Mapped[int] = mapped_column(primary_key=True)
    coupon_id: Mapped[int] = mapped_column(ForeignKey("coupon_codes.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    redeemed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    coupon: Mapped["CouponCode"] = relationship("CouponCode", back_populates="redemptions")
    user: Mapped["User"] = relationship("User")


class GroupLastSeen(Base):
    """Tracks the last time a user read a group's feed — used for unread counts."""
    __tablename__ = "group_last_seen"
    __table_args__ = (UniqueConstraint("user_id", "group_id", name="uq_group_last_seen"),)

    id:           Mapped[int]      = mapped_column(primary_key=True)
    user_id:      Mapped[int]      = mapped_column(ForeignKey("users.id"), index=True)
    group_id:     Mapped[int]      = mapped_column(Integer, index=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class LabelAnnotation(Base):
    """
    User-confirmed intent labels — ground truth for retraining the ONNX classifier.

    Populated two ways:
      1. user_correction: user moves a message to a different bucket in the iOS app
         (PATCH /api/messages/{id}/relabel)
      2. manual: bulk-imported from the seed training data

    These are exported via GET /api/annotations/export and fed into retrain.py
    in the intent-classifier-poc repo to produce updated ONNX weights.
    """
    __tablename__ = "label_annotations"

    id:         Mapped[int]      = mapped_column(primary_key=True)
    user_id:    Mapped[int]      = mapped_column(ForeignKey("users.id"), index=True)
    message_id: Mapped[Optional[int]] = mapped_column(ForeignKey("messages.id"), nullable=True, index=True)
    text:       Mapped[str]      = mapped_column(Text, nullable=False)
    label:      Mapped[str]      = mapped_column(String(50), nullable=False)
    # "user_correction" | "manual"
    source:     Mapped[str]      = mapped_column(String(30), default="user_correction")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    user: Mapped["User"] = relationship("User")


class IAPTransaction(Base):
    """Apple IAP transaction record — links original_transaction_id → user for webhook lookup."""
    __tablename__ = "iap_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    transaction_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    original_transaction_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    product_id: Mapped[str] = mapped_column(String(100), nullable=False)
    environment: Mapped[str] = mapped_column(String(20), default="Production")  # Production | Sandbox
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PaymentOrder(Base):
    """Razorpay payment orders — one row per checkout attempt."""
    __tablename__ = "payment_orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    razorpay_order_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    razorpay_payment_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    plan: Mapped[str] = mapped_column(String(20), nullable=False)    # "monthly" | "annual"
    amount: Mapped[int] = mapped_column(Integer, nullable=False)      # paise
    status: Mapped[str] = mapped_column(String(20), default="created")  # created | paid | failed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class StoredImage(Base):
    """Binary image storage — no external CDN needed."""
    __tablename__ = "stored_images"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    data: Mapped[bytes] = mapped_column(LargeBinary)
    mime_type: Mapped[str] = mapped_column(String(50), default="image/jpeg")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship("User")


class AnalyticsEvent(Base):
    """First-party product analytics — one row per tracked client event.
    Powers the activation funnel + per-user journey readouts in /admin.
    Ingested via POST /api/events (batched). Works logged-in or anonymous —
    anon_id correlates pre-login events (app_open, signup steps) to a device."""
    __tablename__ = "analytics_events"

    id:          Mapped[int]            = mapped_column(primary_key=True)
    user_id:     Mapped[Optional[int]]  = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    anon_id:     Mapped[Optional[str]]  = mapped_column(String(64), nullable=True, index=True)   # device id (pre-login)
    session_id:  Mapped[Optional[str]]  = mapped_column(String(64), nullable=True, index=True)
    event:       Mapped[str]            = mapped_column(String(60), nullable=False, index=True)
    props:       Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    platform:    Mapped[Optional[str]]  = mapped_column(String(20), nullable=True)               # "ios" | "web"
    app_version: Mapped[Optional[str]]  = mapped_column(String(20), nullable=True)
    client_ts:   Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)             # event time on device
    created_at:  Mapped[datetime]       = mapped_column(DateTime, default=datetime.utcnow, index=True)


# Database initialization
async def init_db():
    """Create all tables and apply safe column migrations."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Safe migrations for columns added after initial deploy
        from sqlalchemy import text as sa_text
        migrations = [
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_pro BOOLEAN DEFAULT FALSE",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS active_group_id INTEGER",
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS group_id INTEGER",
            "ALTER TABLE messages ADD COLUMN IF NOT EXISTS assigned_to_user_id INTEGER",
            "CREATE TABLE IF NOT EXISTS group_last_seen (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), group_id INTEGER, last_seen_at TIMESTAMP DEFAULT NOW(), UNIQUE(user_id, group_id))",
            # Performance indexes
            "CREATE INDEX IF NOT EXISTS idx_messages_user_created ON messages(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_messages_group_created ON messages(group_id, created_at DESC) WHERE group_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_messages_assigned ON messages(assigned_to_user_id) WHERE assigned_to_user_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_group_last_seen_lookup ON group_last_seen(user_id, group_id)",
            "ALTER TABLE otp_verifications ALTER COLUMN otp_code TYPE VARCHAR(20)",
            "ALTER TABLE users ALTER COLUMN phone_number TYPE VARCHAR(50)",
            "CREATE TABLE IF NOT EXISTS label_annotations (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id), message_id INTEGER REFERENCES messages(id), text TEXT NOT NULL, label VARCHAR(50) NOT NULL, source VARCHAR(30) DEFAULT 'user_correction', created_at TIMESTAMP DEFAULT NOW())",
            "CREATE INDEX IF NOT EXISTS idx_label_annotations_user ON label_annotations(user_id)",
            "ALTER TABLE users ALTER COLUMN occupation DROP NOT NULL",
            "ALTER TABLE pro_account_members ALTER COLUMN phone_number TYPE VARCHAR(100)",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS google_uid VARCHAR(128)",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS apple_uid VARCHAR(128)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_uid ON users(google_uid) WHERE google_uid IS NOT NULL",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_apple_uid ON users(apple_uid) WHERE apple_uid IS NOT NULL",
            "CREATE TABLE IF NOT EXISTS payment_orders (id SERIAL PRIMARY KEY, razorpay_order_id VARCHAR(100) UNIQUE NOT NULL, razorpay_payment_id VARCHAR(100), user_id INTEGER REFERENCES users(id) NOT NULL, plan VARCHAR(20) NOT NULL, amount INTEGER NOT NULL, status VARCHAR(20) DEFAULT 'created', created_at TIMESTAMP DEFAULT NOW())",
            "CREATE INDEX IF NOT EXISTS idx_payment_orders_user ON payment_orders(user_id)",
            "CREATE TABLE IF NOT EXISTS iap_transactions (id SERIAL PRIMARY KEY, user_id INTEGER REFERENCES users(id) NOT NULL, transaction_id VARCHAR(100) UNIQUE NOT NULL, original_transaction_id VARCHAR(100) NOT NULL, product_id VARCHAR(100) NOT NULL, environment VARCHAR(20) DEFAULT 'Production', expires_at TIMESTAMP, created_at TIMESTAMP DEFAULT NOW())",
            "CREATE INDEX IF NOT EXISTS idx_iap_transactions_original ON iap_transactions(original_transaction_id)",
            "CREATE INDEX IF NOT EXISTS idx_iap_transactions_user ON iap_transactions(user_id)",
            # Self-contained group membership (WhatsApp-style): per-group join link + cap
            "ALTER TABLE groups ADD COLUMN IF NOT EXISTS invite_token VARCHAR(64)",
            "ALTER TABLE groups ADD COLUMN IF NOT EXISTS max_members INTEGER DEFAULT 10",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_groups_invite_token ON groups(invite_token) WHERE invite_token IS NOT NULL",
            # Group profile photo (WhatsApp-style avatar)
            "ALTER TABLE groups ADD COLUMN IF NOT EXISTS photo_url VARCHAR(500)",
        ]
        for stmt in migrations:
            try:
                await conn.execute(sa_text(stmt))
            except Exception as e:
                print(f"[migration] skipped: {e}")

        # Backfill invite_token for any group created before the column existed.
        try:
            import secrets
            rows = (await conn.execute(
                sa_text("SELECT id FROM groups WHERE invite_token IS NULL")
            )).fetchall()
            for (gid,) in rows:
                await conn.execute(
                    sa_text("UPDATE groups SET invite_token = :tok WHERE id = :gid"),
                    {"tok": secrets.token_urlsafe(24), "gid": gid},
                )
        except Exception as e:
            print(f"[migration] invite_token backfill skipped: {e}")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session"""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()
