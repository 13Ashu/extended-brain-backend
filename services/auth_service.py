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

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from database import User, OTPVerification, get_db
from messaging_interface import MessagingClient
from config import Config

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
            "exp": datetime.utcnow() + timedelta(days=7)
        }

        return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    async def send_otp(self, phone_number: str, db: AsyncSession) -> dict:

        if not Config.ENABLE_OTP:
            return {
                "success": True,
                "message": "OTP disabled (dev mode)",
                "expires_in": 0
            }
        try:
            otp_code = self.generate_otp()
            expires_at = datetime.utcnow() + timedelta(
                minutes=self.OTP_EXPIRY_MINUTES
            )

            logger.info(f"Generated OTP for {phone_number}: {otp_code}")

            existing = await db.execute(
                select(OTPVerification)
                .where(OTPVerification.phone_number == phone_number)
            )

            for otp in existing.scalars():
                await db.delete(otp)

            otp_record = OTPVerification(
                phone_number=phone_number,
                otp_code=otp_code,
                expires_at=expires_at
            )

            db.add(otp_record)
            await db.commit()

            message = (
                f"🧠 Extended Brain Verification\n\n"
                f"Your OTP code is: {otp_code}\n\n"
                f"This code will expire in {self.OTP_EXPIRY_MINUTES} minutes.\n"
                f"Do not share this code with anyone."
            )

            try:
                await self.messaging_client.send_message(phone_number, message)
            except Exception as e:
                logger.warning(f"Failed to send OTP: {e}")

            return {
                "success": True,
                "message": "OTP sent successfully",
                "expires_in": self.OTP_EXPIRY_MINUTES
            }

        except Exception as e:
            logger.error(f"Error sending OTP: {e}")
            return {
                "success": False,
                "message": str(e)
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

            otp_record = result.scalar_one_or_none()

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
        phone_number: str,
        password: str,
        db: AsyncSession
    ) -> dict:
        try:
            result = await db.execute(
                select(User).where(User.phone_number == phone_number)
            )

            user = result.scalar_one_or_none()

            if not user:
                return {
                    "success": False,
                    "message": "Invalid credentials"
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
        occupation: str,
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
            occupation=occupation,
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
