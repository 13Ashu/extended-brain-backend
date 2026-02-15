"""
Authentication Service
Handles OTP generation, verification, and user authentication
"""

import random
import string
from datetime import datetime, timedelta
from typing import Optional
import hashlib
import secrets
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import User, OTPVerification


class AuthService:
    """Authentication service for OTP and user management"""
    
    OTP_EXPIRY_MINUTES = 10
    MAX_OTP_ATTEMPTS = 5
    
    def __init__(self, whatsapp_client=None):
        """Initialize with optional WhatsApp client for sending OTP"""
        self.whatsapp_client = whatsapp_client
    
    @staticmethod
    def generate_otp(length: int = 6) -> str:
        """Generate a random OTP"""
        return ''.join(random.choices(string.digits, k=length))
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using SHA256 with salt"""
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}${hashed}"
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hashed = password_hash.split('$')
            test_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return test_hash == hashed
        except:
            return False
    
    async def send_otp(
        self,
        phone_number: str,
        db: AsyncSession
    ) -> dict:
        """
        Generate and send OTP to phone number
        
        Returns:
            dict: {'success': bool, 'message': str}
        """
        try:
            # Generate OTP
            otp_code = self.generate_otp()
            expires_at = datetime.utcnow() + timedelta(minutes=self.OTP_EXPIRY_MINUTES)
            
            # Delete any existing OTPs for this number
            await db.execute(
                select(OTPVerification)
                .where(OTPVerification.phone_number == phone_number)
            )
            existing = await db.execute(
                select(OTPVerification)
                .where(OTPVerification.phone_number == phone_number)
            )
            for otp in existing.scalars():
                await db.delete(otp)
            
            # Create new OTP record
            otp_record = OTPVerification(
                phone_number=phone_number,
                otp_code=otp_code,
                expires_at=expires_at
            )
            db.add(otp_record)
            await db.commit()
            
            # Send OTP via WhatsApp
            if self.whatsapp_client:
                message = (
                    f"🧠 *Extended Brain Verification*\n\n"
                    f"Your OTP code is: *{otp_code}*\n\n"
                    f"This code will expire in {self.OTP_EXPIRY_MINUTES} minutes.\n"
                    f"Do not share this code with anyone."
                )
                await self.whatsapp_client.send_message(phone_number, message)
            else:
                # For development/testing - log the OTP
                print(f"📱 OTP for {phone_number}: {otp_code}")
            
            return {
                'success': True,
                'message': f'OTP sent to {phone_number}',
                'expires_in': self.OTP_EXPIRY_MINUTES
            }
            
        except Exception as e:
            print(f"Error sending OTP: {e}")
            return {
                'success': False,
                'message': f'Failed to send OTP: {str(e)}'
            }
    
    async def verify_otp(
        self,
        phone_number: str,
        otp_code: str,
        db: AsyncSession
    ) -> dict:
        """
        Verify OTP code
        
        Returns:
            dict: {'success': bool, 'message': str, 'verified': bool}
        """
        try:
            # Find the most recent OTP for this number
            result = await db.execute(
                select(OTPVerification)
                .where(OTPVerification.phone_number == phone_number)
                .where(OTPVerification.is_verified == False)
                .order_by(OTPVerification.created_at.desc())
            )
            otp_record = result.scalar_one_or_none()
            
            if not otp_record:
                return {
                    'success': False,
                    'message': 'No OTP found. Please request a new one.',
                    'verified': False
                }
            
            # Check if expired
            if datetime.utcnow() > otp_record.expires_at:
                return {
                    'success': False,
                    'message': 'OTP has expired. Please request a new one.',
                    'verified': False
                }
            
            # Check max attempts
            if otp_record.attempts >= self.MAX_OTP_ATTEMPTS:
                return {
                    'success': False,
                    'message': 'Too many failed attempts. Please request a new OTP.',
                    'verified': False
                }
            
            # Verify OTP
            if otp_record.otp_code == otp_code:
                otp_record.is_verified = True
                await db.commit()
                
                return {
                    'success': True,
                    'message': 'OTP verified successfully',
                    'verified': True
                }
            else:
                # Increment attempts
                otp_record.attempts += 1
                await db.commit()
                
                remaining = self.MAX_OTP_ATTEMPTS - otp_record.attempts
                return {
                    'success': False,
                    'message': f'Invalid OTP. {remaining} attempts remaining.',
                    'verified': False
                }
                
        except Exception as e:
            print(f"Error verifying OTP: {e}")
            return {
                'success': False,
                'message': f'Failed to verify OTP: {str(e)}',
                'verified': False
            }
    
    async def check_otp_verified(
        self,
        phone_number: str,
        db: AsyncSession
    ) -> bool:
        """Check if phone number has been verified via OTP"""
        result = await db.execute(
            select(OTPVerification)
            .where(OTPVerification.phone_number == phone_number)
            .where(OTPVerification.is_verified == True)
            .order_by(OTPVerification.created_at.desc())
        )
        otp_record = result.scalar_one_or_none()
        
        if not otp_record:
            return False
        
        # Check if verification was recent (within 1 hour)
        time_since_verification = datetime.utcnow() - otp_record.created_at
        return time_since_verification.total_seconds() < 3600
    
    async def register_user(
        self,
        phone_number: str,
        name: str,
        email: str,
        age: int,
        occupation: str,
        password: str,
        timezone: str,
        db: AsyncSession
    ) -> dict:
        """
        Register a new user after OTP verification
        
        Returns:
            dict: {'success': bool, 'message': str, 'user_id': int}
        """
        try:
            # Check if phone is verified
            is_verified = await self.check_otp_verified(phone_number, db)
            if not is_verified:
                return {
                    'success': False,
                    'message': 'Phone number not verified. Please verify OTP first.'
                }
            
            # Check if user already exists
            result = await db.execute(
                select(User).where(
                    (User.phone_number == phone_number) | (User.email == email)
                )
            )
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                return {
                    'success': False,
                    'message': 'User with this phone number or email already exists'
                }
            
            # Hash password
            password_hash = self.hash_password(password)
            
            # Create new user
            new_user = User(
                phone_number=phone_number,
                email=email,
                name=name,
                age=age,
                occupation=occupation,
                password_hash=password_hash,
                timezone=timezone,
                is_verified=True,
                is_active=True
            )
            
            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)
            
            # Send welcome message via WhatsApp
            if self.whatsapp_client:
                welcome_message = (
                    f"🎉 *Welcome to Extended Brain, {name}!*\n\n"
                    f"Your second brain is ready to capture your thoughts, ideas, and content.\n\n"
                    f"*How to use:*\n"
                    f"📝 Send any text, image, or audio\n"
                    f"🔍 Search with: 'search: your query'\n"
                    f"📁 View categories: 'categories'\n\n"
                    f"Start sending messages and I'll organize everything for you!"
                )
                await self.whatsapp_client.send_message(phone_number, welcome_message)
            
            return {
                'success': True,
                'message': 'User registered successfully',
                'user_id': new_user.id
            }
            
        except Exception as e:
            print(f"Error registering user: {e}")
            await db.rollback()
            return {
                'success': False,
                'message': f'Failed to register user: {str(e)}'
            }
    
    async def login_user(
        self,
        phone_number: str,
        password: str,
        db: AsyncSession
    ) -> dict:
        """
        Authenticate user login
        
        Returns:
            dict: {'success': bool, 'message': str, 'user': User}
        """
        try:
            # Find user
            result = await db.execute(
                select(User).where(User.phone_number == phone_number)
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return {
                    'success': False,
                    'message': 'Invalid phone number or password'
                }
            
            # Verify password
            if not self.verify_password(password, user.password_hash):
                return {
                    'success': False,
                    'message': 'Invalid phone number or password'
                }
            
            # Update last login
            user.last_login = datetime.utcnow()
            await db.commit()
            
            return {
                'success': True,
                'message': 'Login successful',
                'user': {
                    'id': user.id,
                    'phone_number': user.phone_number,
                    'name': user.name,
                    'email': user.email,
                    'timezone': user.timezone
                }
            }
            
        except Exception as e:
            print(f"Error during login: {e}")
            return {
                'success': False,
                'message': f'Login failed: {str(e)}'
            }
