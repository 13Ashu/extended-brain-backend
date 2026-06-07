"""
Authentication Service
Handles OTP generation, verification, and user authentication
Updated for multi-platform support (WhatsApp/Telegram)
"""

import random
import string
from datetime import datetime, timedelta
from typing import Optional
import hashlib
import secrets
import os

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from database import User, OTPVerification, get_db
from messaging_interface import MessagingClient
from config import Config
from services.sms_service import sms_service

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key")
ALGORITHM = "HS256"


class AuthService:
    """Authentication service for OTP and user management"""

    OTP_EXPIRY_MINUTES = 10
    MAX_OTP_ATTEMPTS = 5
    RESEND_COOLDOWN_SECONDS = Config.OTP_RESEND_COOLDOWN_SECONDS
    MAX_OTP_PER_DAY = Config.OTP_MAX_PER_DAY

    def __init__(self, messaging_client: MessagingClient):
        self.messaging_client = messaging_client

    @staticmethod
    def generate_otp(length: int = 6) -> str:
        return ''.join(random.choices(string.digits, k=length))

    @staticmethod
    def hash_password(password: str) -> str:
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}${hashed}"

    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        try:
            salt, hashed = password_hash.split('$')
            test_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return test_hash == hashed
        except Exception:
            return False

    def create_access_token(self, user_id: int) -> str:
        payload = {
            "sub": str(user_id),  # store user ID
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(days=30)
        }

        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    def create_oauth_session_token(self, provider: str, uid: str, email: str, name: str) -> str:
        payload = {
            "sub": "oauth_session",
            "provider": provider,
            "uid": uid,
            "email": email,
            "name": name,
            "exp": datetime.utcnow() + timedelta(hours=24),
        }
        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    def decode_oauth_session_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("sub") != "oauth_session":
                raise ValueError("Invalid session token type")
            return payload
        except JWTError:
            raise ValueError("Invalid or expired session token")

    async def send_otp(self, phone_number: str, db: AsyncSession) -> dict:

        if not Config.ENABLE_OTP:
            return {
                "success": True,
                "message": "OTP disabled (dev mode)",
                "expires_in": 0
            }
        try:
            now = datetime.utcnow()
            day_ago = now - timedelta(hours=24)

            # Pull this phone's OTPs from the last 24h for anti-pumping checks.
            recent_result = await db.execute(
                select(OTPVerification)
                .where(OTPVerification.phone_number == phone_number)
                .where(OTPVerification.created_at > day_ago)
                .order_by(OTPVerification.created_at.desc())
            )
            recent = recent_result.scalars().all()

            # Resend cooldown — block rapid re-requests (we pay per SMS).
            if recent:
                since_last = (now - recent[0].created_at).total_seconds()
                if since_last < self.RESEND_COOLDOWN_SECONDS:
                    wait = int(self.RESEND_COOLDOWN_SECONDS - since_last) + 1
                    return {
                        "success": False,
                        "message": f"Please wait {wait}s before requesting another code.",
                        "retry_after": wait,
                    }

            # Daily cap per phone number.
            if len(recent) >= self.MAX_OTP_PER_DAY:
                return {
                    "success": False,
                    "message": "Too many OTP requests today. Please try again later.",
                }

            otp_code = self.generate_otp()
            expires_at = now + timedelta(minutes=self.OTP_EXPIRY_MINUTES)

            if Config.DEBUG:
                logger.info(f"Generated OTP for {phone_number}: {otp_code}")
            else:
                logger.info("Generated OTP for ***{}", phone_number[-4:])

            # Invalidate any still-valid previous code (one active code at a time)
            # but keep the rows so the 24h rate-limit counter stays accurate.
            for r in recent:
                if not r.is_verified and r.expires_at > now:
                    r.expires_at = now

            # Hard-delete only truly-old rows (>24h) to keep the table bounded.
            await db.execute(
                delete(OTPVerification)
                .where(OTPVerification.phone_number == phone_number)
                .where(OTPVerification.created_at <= day_ago)
            )

            otp_record = OTPVerification(
                phone_number=phone_number,
                otp_code=otp_code,
                expires_at=expires_at,
            )
            db.add(otp_record)
            await db.commit()

            # Deliver. Prefer MSG91 SMS when configured (production OTP transport);
            # otherwise fall back to the legacy messaging client (Telegram) so the
            # existing /send-otp path keeps working until MSG91 is switched on.
            if sms_service.is_configured:
                # On failure, roll back the stored code so a user who never
                # received an SMS can't be left with a "valid" OTP.
                try:
                    await sms_service.send_otp(phone_number, otp_code)
                except Exception as e:
                    logger.error(f"Failed to send OTP SMS: {e}")
                    await db.delete(otp_record)
                    await db.commit()
                    return {
                        "success": False,
                        "message": "Could not send verification code. Please try again.",
                    }
            else:
                # Legacy fallback — kept ready; lenient (matches prior behaviour).
                message = (
                    f"Extended Minds verification\n\n"
                    f"Your OTP code is: {otp_code}\n\n"
                    f"This code expires in {self.OTP_EXPIRY_MINUTES} minutes.\n"
                    f"Do not share it with anyone."
                )
                try:
                    await self.messaging_client.send_message(phone_number, message)
                except Exception as e:
                    logger.warning(f"Failed to send OTP via messaging client: {e}")

            return {
                "success": True,
                "message": "OTP sent successfully",
                "expires_in": self.OTP_EXPIRY_MINUTES
            }

        except Exception as e:
            logger.error(f"Error sending OTP: {e}")
            return {
                "success": False,
                "message": "Could not send verification code.",
            }

    async def verify_otp(
        self,
        phone_number: str,
        otp_code: str,
        db: AsyncSession
    ) -> dict:
        
        if not Config.ENABLE_OTP:
            return {
                "success": True,
                "message": "OTP verification skipped (dev mode)",
                "verified": True
            }
    
        try:
            result = await db.execute(
                select(OTPVerification)
                .where(OTPVerification.phone_number == phone_number)
                .where(OTPVerification.is_verified == False)
                .order_by(OTPVerification.created_at.desc())
            )

            # send_otp() now keeps superseded rows for rate-limiting, so there
            # can be several unverified rows — the newest is the active code.
            otp_record = result.scalars().first()

            if not otp_record:
                return {
                    "success": False,
                    "message": "No OTP found",
                    "verified": False
                }

            if datetime.utcnow() > otp_record.expires_at:
                return {
                    "success": False,
                    "message": "OTP expired",
                    "verified": False
                }

            if otp_record.attempts >= self.MAX_OTP_ATTEMPTS:
                return {
                    "success": False,
                    "message": "Too many attempts",
                    "verified": False
                }

            if otp_record.otp_code == otp_code:
                otp_record.is_verified = True
                await db.commit()

                return {
                    "success": True,
                    "message": "OTP verified",
                    "verified": True
                }

            otp_record.attempts += 1
            await db.commit()

            remaining = self.MAX_OTP_ATTEMPTS - otp_record.attempts

            return {
                "success": False,
                "message": f"Invalid OTP. {remaining} attempts remaining.",
                "verified": False
            }

        except Exception as e:
            logger.error(f"OTP verification error: {e}")
            return {
                "success": False,
                "message": str(e),
                "verified": False
            }

    async def login_user(
        self,
        password: str,
        db: AsyncSession,
        email: str = None,
        phone_number: str = None,
    ) -> dict:
        try:
            user = None
            if email:
                user = await db.scalar(select(User).where(User.email == email))
            if not user and email:
                user = await db.scalar(select(User).where(User.phone_number == email))
            if not user and phone_number:
                user = await db.scalar(select(User).where(User.phone_number == phone_number))

            if not user:
                return {
                    "success": False,
                    "message": "Invalid credentials"
                }

            if user.password_hash == "google_oauth_no_password":
                return {
                    "success": False,
                    "message": "This account was created with Google Sign-In. Please use the 'Sign in with Google' button instead."
                }
            if user.password_hash == "apple_oauth_no_password":
                return {
                    "success": False,
                    "message": "This account was created with Apple Sign-In. Please use the 'Sign in with Apple' button instead."
                }

            if not self.verify_password(password, user.password_hash):
                return {
                    "success": False,
                    "message": "Invalid credentials"
                }

            user.last_login = datetime.utcnow()
            await db.commit()

            access_token = self.create_access_token(user.id)

            return {
                "success": True,
                "message": "Login successful",
                "data": {
                    "access_token": access_token,
                    "user": {
                        "id": user.id,
                        "phone_number": user.phone_number,
                        "name": user.name,
                        "email": user.email,
                        "timezone": user.timezone
                    }
                }
            }

        except Exception as e:
            logger.error(f"Login error: {e}")
            return {
                "success": False,
                "message": str(e)
            }

    async def register_user(
        self,
        phone_number: str,
        name: str,
        email: str,
        age: int,
        password: str,
        timezone: str = "UTC",
        db: AsyncSession = None
    ) -> dict:
        """
        Register a new user.
        OTP verification is optional based on Config.ENABLE_OTP.
        """

        # Check if user already exists
        result = await db.execute(
            select(User).where(User.phone_number == phone_number)
        )
        existing_user = result.scalar_one_or_none()
        if existing_user:
            return {"success": False, "message": "User already exists"}

        # If OTP is enabled, verify it first
        if Config.ENABLE_OTP:
            otp_result = await db.execute(
                select(OTPVerification)
                .where(OTPVerification.phone_number == phone_number)
                .where(OTPVerification.is_verified == True)
            )
            otp_verified = otp_result.scalar_one_or_none()
            if not otp_verified:
                return {"success": False, "message": "OTP verification required"}

        # Hash the password
        password_hash = self.hash_password(password)

        # Create the new user
        new_user = User(
            name=name,
            email=email,
            age=age,
            phone_number=phone_number,
            password_hash=password_hash,
            timezone=timezone,
            created_at=datetime.utcnow(),
            last_login=None
        )

        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        # Generate access token
        access_token = self.create_access_token(new_user.id)

        return {
            "success": True,
            "message": "User registered successfully",
            "data": {
                "access_token": access_token,
                "user": {
                    "id": new_user.id,
                    "phone_number": new_user.phone_number,
                    "name": new_user.name,
                    "email": new_user.email,
                    "timezone": new_user.timezone
                }
            }
        }

    async def reset_password(self, phone_number: str, new_password: str, db: AsyncSession):
        user = await db.scalar(select(User).where(User.phone_number == phone_number))
        if not user:
            return {"success": False, "message": "No account found with this phone number"}
        
        # Use the same hash_password as register_user and login_user
        new_hash = self.hash_password(new_password)
        await db.execute(
            update(User)
            .where(User.phone_number == phone_number)
            .values(password_hash=new_hash)
        )
        await db.commit()
        
        return {"success": True, "message": "Password reset successfully"}

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Decode JWT token and return current user
    """

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")

        if user_id is None:
            raise credentials_exception

    except JWTError:
        raise credentials_exception

    result = await db.execute(
        select(User).where(User.id == int(user_id))
    )

    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    return user
