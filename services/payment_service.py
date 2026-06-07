"""
Razorpay payment integration for Pro subscriptions.
Uses httpx directly (async-safe) — no razorpay SDK needed.
"""

import hashlib
import hmac
import json
from datetime import datetime, timedelta
from typing import Optional

import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from config import Config
from database import PaymentOrder, ProAccount, User

RAZORPAY_API_BASE = "https://api.razorpay.com/v1"

PLAN_AMOUNTS = {
    "monthly": 29900,    # ₹299 in paise
    "annual":  199900,   # ₹1,999 in paise
}

PLAN_DAYS = {
    "monthly": 31,
    "annual":  365,
}


class PaymentService:

    async def create_order(self, user: User, plan: str, db: AsyncSession) -> dict:
        if plan not in PLAN_AMOUNTS:
            return {"success": False, "message": "Invalid plan. Choose 'monthly' or 'annual'."}

        key_id = Config.RAZORPAY_KEY_ID
        key_secret = Config.RAZORPAY_KEY_SECRET
        if not key_id or not key_secret:
            return {"success": False, "message": "Payments not configured yet."}

        amount = PLAN_AMOUNTS[plan]
        receipt = f"em_{user.id}_{plan}_{int(datetime.utcnow().timestamp())}"

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{RAZORPAY_API_BASE}/orders",
                auth=(key_id, key_secret),
                json={"amount": amount, "currency": "INR", "receipt": receipt},
                timeout=15,
            )

        if resp.status_code != 200:
            return {"success": False, "message": "Could not create payment order. Try again."}

        data = resp.json()
        order = PaymentOrder(
            razorpay_order_id=data["id"],
            user_id=user.id,
            plan=plan,
            amount=amount,
            status="created",
        )
        db.add(order)
        await db.commit()

        return {
            "success": True,
            "order_id": data["id"],
            "amount": amount,
            "currency": "INR",
            "key_id": key_id,
            "plan": plan,
        }

    async def verify_and_activate(
        self,
        user: User,
        order_id: str,
        payment_id: str,
        signature: str,
        db: AsyncSession,
    ) -> dict:
        key_secret = Config.RAZORPAY_KEY_SECRET
        if not key_secret:
            return {"success": False, "message": "Payments not configured."}

        # HMAC-SHA256 of "{order_id}|{payment_id}" with key secret
        expected = hmac.new(
            key_secret.encode(),
            f"{order_id}|{payment_id}".encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(expected, signature):
            return {"success": False, "message": "Payment verification failed."}

        order = await db.scalar(
            select(PaymentOrder).where(
                PaymentOrder.razorpay_order_id == order_id,
                PaymentOrder.user_id == user.id,
            )
        )
        if not order:
            return {"success": False, "message": "Order not found."}
        if order.status == "paid":
            return {"success": False, "message": "Order already processed."}

        plan = order.plan
        expires_at = datetime.utcnow() + timedelta(days=PLAN_DAYS[plan])

        pro = await db.scalar(select(ProAccount).where(ProAccount.owner_id == user.id))
        if pro:
            pro.expires_at = expires_at
            pro.plan_type = f"paid_{plan}"
        else:
            db.add(ProAccount(
                owner_id=user.id,
                plan_type=f"paid_{plan}",
                max_members=6,
                expires_at=expires_at,
            ))

        user.is_pro = True
        order.status = "paid"
        order.razorpay_payment_id = payment_id
        await db.commit()

        label = "monthly" if plan == "monthly" else "annual"
        return {
            "success": True,
            "message": f"Pro {label} activated!",
            "expires_at": expires_at.isoformat(),
            "plan": plan,
        }

    async def handle_webhook(self, body: bytes, signature: str, db: AsyncSession) -> bool:
        webhook_secret = Config.RAZORPAY_WEBHOOK_SECRET
        if not webhook_secret:
            return False

        expected = hmac.new(
            webhook_secret.encode(), body, hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected, signature):
            return False

        event = json.loads(body)
        if event.get("event") == "payment.captured":
            payment = event.get("payload", {}).get("payment", {}).get("entity", {})
            order_id = payment.get("order_id")
            if order_id:
                order = await db.scalar(
                    select(PaymentOrder).where(PaymentOrder.razorpay_order_id == order_id)
                )
                if order and order.status != "paid":
                    order.status = "paid"
                    await db.commit()
        return True


payment_service = PaymentService()
