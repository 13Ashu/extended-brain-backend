"""
Coupon / promo code service.
Handles creation, validation, and redemption of coupon codes for Pro access.
"""
from __future__ import annotations

import secrets
import string
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from config import Config
from database import CouponCode, CouponRedemption, ProAccount, User


class CouponService:

    # ── Admin: create ─────────────────────────────────────────────

    async def create_coupon(
        self,
        code: Optional[str],
        description: Optional[str],
        discount_type: str,          # 'free' | 'percent' | 'fixed'
        discount_value: int,         # 100 for free, % or ₹ otherwise
        duration_days: Optional[int],
        max_uses: Optional[int],
        expires_in_days: Optional[int],
        db: AsyncSession,
    ) -> Dict[str, Any]:
        # Auto-generate code if not provided
        if not code:
            code = self._generate_code()
        else:
            code = code.strip().upper()

        # Check uniqueness
        existing = await db.scalar(select(CouponCode).where(CouponCode.code == code))
        if existing:
            return {"success": False, "message": f"Code '{code}' already exists."}

        expires_at = (
            datetime.utcnow() + timedelta(days=expires_in_days)
            if expires_in_days else None
        )

        coupon = CouponCode(
            code=code,
            description=description,
            discount_type=discount_type,
            discount_value=discount_value,
            duration_days=duration_days,
            max_uses=max_uses,
            expires_at=expires_at,
        )
        db.add(coupon)
        await db.commit()
        return {"success": True, "code": code, "coupon_id": coupon.id}

    # ── Validate (no side effects) ────────────────────────────────

    async def validate(self, code: str, user: User, db: AsyncSession) -> Dict[str, Any]:
        coupon = await db.scalar(
            select(CouponCode).where(CouponCode.code == code.strip().upper())
        )
        if not coupon:
            return {"valid": False, "message": "Invalid coupon code."}
        if not coupon.is_active:
            return {"valid": False, "message": "This coupon is no longer active."}
        if coupon.expires_at and coupon.expires_at < datetime.utcnow():
            return {"valid": False, "message": "This coupon has expired."}
        if coupon.max_uses and coupon.uses_count >= coupon.max_uses:
            return {"valid": False, "message": "This coupon has reached its usage limit."}

        # Check if user already redeemed
        already = await db.scalar(
            select(CouponRedemption).where(
                CouponRedemption.coupon_id == coupon.id,
                CouponRedemption.user_id == user.id,
            )
        )
        if already:
            return {"valid": False, "message": "You've already used this coupon."}

        return {
            "valid": True,
            "coupon_id": coupon.id,
            "discount_type": coupon.discount_type,
            "discount_value": coupon.discount_value,
            "duration_days": coupon.duration_days,
            "description": coupon.description or "",
        }

    # ── Redeem ────────────────────────────────────────────────────

    async def redeem(self, code: str, user: User, db: AsyncSession) -> Dict[str, Any]:
        result = await self.validate(code, user, db)
        if not result["valid"]:
            return {"success": False, "message": result["message"]}

        coupon = await db.scalar(
            select(CouponCode).where(CouponCode.code == code.strip().upper())
        )

        # Record redemption
        db.add(CouponRedemption(coupon_id=coupon.id, user_id=user.id))
        coupon.uses_count += 1

        # Activate Pro
        user.is_pro = True

        # Create/update Pro account
        acct = await db.scalar(select(ProAccount).where(ProAccount.owner_id == user.id))
        if not acct:
            acct = ProAccount(owner_id=user.id)
            db.add(acct)
            await db.flush()

        # Set expiry on pro account
        if coupon.duration_days:
            acct.expires_at = datetime.utcnow() + timedelta(days=coupon.duration_days)
        else:
            acct.expires_at = None  # forever

        await db.commit()

        duration_str = (
            f"{coupon.duration_days} days"
            if coupon.duration_days else "unlimited"
        )
        return {
            "success": True,
            "message": f"Pro activated for {duration_str}! 🎉",
            "duration_days": coupon.duration_days,
            "discount_type": coupon.discount_type,
        }

    # ── Admin: list all coupons ───────────────────────────────────

    async def list_coupons(self, db: AsyncSession) -> List[Dict]:
        rows = await db.execute(
            select(CouponCode).order_by(CouponCode.created_at.desc())
        )
        coupons = []
        for (c,) in rows.all():
            redemption_rows = await db.execute(
                select(CouponRedemption, User)
                .join(User, User.id == CouponRedemption.user_id)
                .where(CouponRedemption.coupon_id == c.id)
                .order_by(CouponRedemption.redeemed_at.desc())
            )
            redeemers = [
                {"name": u.name, "phone": u.phone_number, "at": r.redeemed_at.isoformat()}
                for r, u in redemption_rows.all()
            ]
            coupons.append({
                "id":             c.id,
                "code":           c.code,
                "description":    c.description or "",
                "discount_type":  c.discount_type,
                "discount_value": c.discount_value,
                "duration_days":  c.duration_days,
                "max_uses":       c.max_uses,
                "uses_count":     c.uses_count,
                "is_active":      c.is_active,
                "expires_at":     c.expires_at.isoformat() if c.expires_at else None,
                "created_at":     c.created_at.isoformat(),
                "redeemers":      redeemers,
            })
        return coupons

    async def deactivate(self, coupon_id: int, db: AsyncSession) -> bool:
        c = await db.get(CouponCode, coupon_id)
        if not c:
            return False
        c.is_active = False
        await db.commit()
        return True

    # ── Founding-member scarcity status (public) ──────────────────
    async def founding_status(self, db: AsyncSession) -> Dict[str, Any]:
        """Live usage of the founding (FOUNDER) coupon, for the scarcity pill.
        `enabled` = display flag (Config.FOUNDING_PILL_ENABLED).
        `active`  = offer still claimable (active, not expired, slots remain).
        UI shows the pill only when enabled AND active. Never raises."""
        code    = (Config.FOUNDING_COUPON_CODE or "").strip().upper()
        enabled = Config.FOUNDING_PILL_ENABLED
        coupon  = (
            await db.scalar(select(CouponCode).where(CouponCode.code == code))
            if code else None
        )
        if not coupon:
            total = Config.FOUNDING_SLOTS_TOTAL
            return {
                "enabled": enabled, "active": False, "code": code or None,
                "slots_total": total, "slots_used": 0, "slots_left": total,
                "expires_at": None,
            }

        total = coupon.max_uses if coupon.max_uses else Config.FOUNDING_SLOTS_TOTAL
        used  = coupon.uses_count or 0
        left  = max(0, total - used)
        not_expired = not (coupon.expires_at and coupon.expires_at < datetime.utcnow())
        return {
            "enabled":     enabled,
            "active":      bool(coupon.is_active and not_expired and left > 0),
            "code":        coupon.code,
            "slots_total": total,
            "slots_used":  used,
            "slots_left":  left,
            "expires_at":  coupon.expires_at.isoformat() if coupon.expires_at else None,
        }

    # ── Helpers ───────────────────────────────────────────────────

    def _generate_code(self, length: int = 8) -> str:
        chars = string.ascii_uppercase + string.digits
        # Remove confusable chars
        chars = chars.replace("O", "").replace("0", "").replace("I", "").replace("1", "")
        return "".join(secrets.choice(chars) for _ in range(length))


coupon_service = CouponService()
