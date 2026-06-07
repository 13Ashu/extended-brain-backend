"""
Apple In-App Purchase verification and subscription management.

Two responsibilities:
  1. /api/iap/verify   — called by iOS after a StoreKit 2 purchase; verifies the
                         transaction with Apple's App Store Server API and activates Pro.
  2. /webhook/apple    — receives App Store Server Notifications V2 for renewals,
                         expirations, refunds, etc., and keeps Pro status in sync.

Key design decisions:
  - JWS signatures are verified against Apple's live JWKS (cached 1 h) to prevent forgery.
  - `original_transaction_id` stored in iap_transactions table is the link between Apple
    notifications (no user JWT) and backend users.
  - Grace-period and billing-retry states keep Pro active — don't punish users whose
    Apple Pay card momentarily declined.
  - All writes are idempotent: re-processing the same transaction is safe.
"""

import base64
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import httpx
from cryptography.hazmat.backends import default_backend
from cryptography.x509 import load_pem_x509_certificate
import jwt as pyjwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import Config
from database import IAPTransaction, ProAccount, User

logger = logging.getLogger(__name__)

# Product ID → (plan label, days of Pro access)
PRODUCT_PLAN = {
    "com.extendedminds.app.pro.monthly": ("monthly", 31),
    "com.extendedminds.app.pro.annual":  ("annual",  365),
}

# Notification types that mean "keep Pro active even though renewal failed"
GRACE_TYPES = {"DID_FAIL_TO_RENEW", "GRACE_PERIOD_EXPIRED"}
# Notification types that explicitly revoke Pro
REVOKE_TYPES = {"EXPIRED", "REFUND", "REVOKE"}
# Notification types that grant / extend Pro
GRANT_TYPES  = {"SUBSCRIBED", "DID_RENEW", "RENEWAL", "OFFER_REDEEMED"}

APPLE_JWKS_URL = "https://appleid.apple.com/auth/keys"
APPLE_API_BASE  = "https://api.storekit.itunes.apple.com"
APPLE_SBOX_BASE = "https://api.storekit-sandbox.itunes.apple.com"


# ─────────────────────────────────────────────────────────────────────────────
# JWS verifier (shared singleton, thread-safe because we only read the cache)
# ─────────────────────────────────────────────────────────────────────────────

class _AppleJWSVerifier:
    """
    Verifies Apple-signed JWS tokens using Apple's live JWKS endpoint.
    Public keys are cached for 1 hour to avoid hammering Apple's CDN.
    """

    def __init__(self) -> None:
        self._key_cache: dict[str, object] = {}   # kid → public-key object
        self._cache_ts: float = 0.0

    async def _refresh_keys(self) -> None:
        async with httpx.AsyncClient(timeout=10) as c:
            resp = await c.get(APPLE_JWKS_URL)
            resp.raise_for_status()
            jwks = resp.json()

        keys: dict[str, object] = {}
        for k in jwks.get("keys", []):
            kid = k.get("kid")
            x5c = k.get("x5c", [])
            if not kid or not x5c:
                continue
            pem = f"-----BEGIN CERTIFICATE-----\n{x5c[0]}\n-----END CERTIFICATE-----"
            cert = load_pem_x509_certificate(pem.encode(), default_backend())
            keys[kid] = cert.public_key()

        self._key_cache = keys
        self._cache_ts  = time.monotonic()
        logger.info("[IAP] Refreshed Apple JWKS (%d keys)", len(keys))

    async def _get_key(self, kid: str) -> object:
        if not self._key_cache or (time.monotonic() - self._cache_ts) > 3600:
            await self._refresh_keys()
        if kid not in self._key_cache:
            # Retry once — Apple may have rotated
            await self._refresh_keys()
        if kid not in self._key_cache:
            raise ValueError(f"Apple public key '{kid}' not found in JWKS")
        return self._key_cache[kid]

    async def verify(self, token: str, *, audience: Optional[str] = None) -> dict:
        """
        Verify an Apple-signed JWS and return the decoded payload dict.
        Raises jwt.DecodeError / ValueError on any failure.
        """
        header = pyjwt.get_unverified_header(token)
        kid    = header.get("kid")
        if not kid:
            raise ValueError("Missing 'kid' in JWS header")

        public_key = await self._get_key(kid)

        options = {"verify_exp": False}   # Apple's notification JWTs use their own expiry
        decode_kwargs: dict = dict(
            algorithms=["ES256"],
            options=options,
        )
        if audience:
            decode_kwargs["audience"] = audience
        else:
            decode_kwargs["options"]["verify_aud"] = False

        payload = pyjwt.decode(token, public_key, **decode_kwargs)
        return payload


_verifier = _AppleJWSVerifier()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safely decode JWS payload without verifying (used only when we've
# already obtained the token from Apple's own authenticated API endpoint).
# ─────────────────────────────────────────────────────────────────────────────

def _decode_jws_payload(jws: str) -> dict:
    parts = jws.split(".")
    if len(parts) != 3:
        raise ValueError("Not a valid JWS")
    padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
    return json.loads(base64.urlsafe_b64decode(padded))


# ─────────────────────────────────────────────────────────────────────────────
# Core IAP service
# ─────────────────────────────────────────────────────────────────────────────

class IAPService:

    # ── App Store Server API JWT ─────────────────────────────────────────────

    def _make_api_jwt(self) -> str:
        """Short-lived JWT (5 min) for App Store Server API calls."""
        if not Config.APPLE_ISSUER_ID or not Config.APPLE_KEY_ID or not Config.APPLE_PRIVATE_KEY:
            raise RuntimeError("Apple IAP credentials (APPLE_ISSUER_ID / APPLE_KEY_ID / APPLE_PRIVATE_KEY) not configured")

        now = int(time.time())
        payload = {
            "iss": Config.APPLE_ISSUER_ID,
            "iat": now,
            "exp": now + 300,              # 5 minutes — well under Apple's 1 h max
            "aud": "appstoreconnect-v1",   # required literal value
            "bid": Config.APNS_BUNDLE_ID,  # reuse existing APNS bundle ID env var
        }
        return pyjwt.encode(
            payload,
            Config.APPLE_PRIVATE_KEY,
            algorithm="ES256",
            headers={"kid": Config.APPLE_KEY_ID},
        )

    # ── Fetch + verify a transaction from Apple ──────────────────────────────

    async def _fetch_signed_transaction(self, transaction_id: str) -> dict:
        """
        Fetch the signed transaction info from Apple's API.
        Tries production first, falls back to sandbox (for TestFlight / development).
        Returns the decoded JWS payload.
        """
        token   = self._make_api_jwt()
        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient(timeout=15) as c:
            for base in (APPLE_API_BASE, APPLE_SBOX_BASE):
                resp = await c.get(f"{base}/inApps/v1/transactions/{transaction_id}", headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    signed = data.get("signedTransactionInfo")
                    if not signed:
                        raise ValueError("Apple API returned no signedTransactionInfo")
                    # Verify signature against Apple's JWKS
                    tx = await _verifier.verify(signed)
                    if tx.get("bundleId") != Config.APNS_BUNDLE_ID:
                        raise ValueError(f"Bundle ID mismatch: {tx.get('bundleId')}")
                    return tx
                if resp.status_code != 404:
                    logger.warning("[IAP] Apple API %s returned %d", base, resp.status_code)

        raise ValueError(f"Transaction {transaction_id} not found in production or sandbox")

    # ── Pro activation helpers ───────────────────────────────────────────────

    async def _activate_pro(
        self,
        user: User,
        product_id: str,
        original_transaction_id: str,
        transaction_id: str,
        expires_ms: Optional[int],
        environment: str,
        db: AsyncSession,
    ) -> datetime:
        """Grant or extend Pro. Returns the new expiry datetime."""
        plan, days = PRODUCT_PLAN.get(product_id, ("monthly", 31))

        # Use Apple's expiry date if present (milliseconds epoch); otherwise estimate
        if expires_ms:
            expires_at = datetime.utcfromtimestamp(expires_ms / 1000)
            # Add a 24-hour buffer so a near-midnight renewal doesn't briefly revoke access
            expires_at += timedelta(hours=24)
        else:
            expires_at = datetime.utcnow() + timedelta(days=days)

        # Upsert ProAccount
        pro = await db.scalar(select(ProAccount).where(ProAccount.owner_id == user.id))
        if pro:
            pro.expires_at  = expires_at
            pro.plan_type   = f"iap_{plan}"
        else:
            db.add(ProAccount(
                owner_id    = user.id,
                plan_type   = f"iap_{plan}",
                max_members = 6,
                expires_at  = expires_at,
            ))

        user.is_pro = True

        # Upsert IAPTransaction for webhook → user lookup later
        existing = await db.scalar(
            select(IAPTransaction).where(IAPTransaction.transaction_id == transaction_id)
        )
        if not existing:
            db.add(IAPTransaction(
                user_id                 = user.id,
                transaction_id          = transaction_id,
                original_transaction_id = original_transaction_id,
                product_id              = product_id,
                environment             = environment,
                expires_at              = expires_at,
            ))

        await db.commit()
        return expires_at

    async def _revoke_pro(self, user: User, db: AsyncSession) -> None:
        user.is_pro = False
        pro = await db.scalar(select(ProAccount).where(ProAccount.owner_id == user.id))
        if pro:
            pro.expires_at = datetime.utcnow()
        await db.commit()

    # ── Public: verify purchase from iOS ────────────────────────────────────

    async def verify_and_activate(
        self,
        user: User,
        transaction_id: str,
        original_transaction_id: str,
        db: AsyncSession,
    ) -> dict:
        """
        Called by POST /api/iap/verify (authenticated endpoint).
        Verifies the transaction with Apple and activates Pro.
        """
        try:
            tx = await self._fetch_signed_transaction(transaction_id)
        except Exception as e:
            logger.error("[IAP] Transaction fetch failed: %s", e)
            return {"success": False, "message": str(e)}

        product_id = tx.get("productId", "")
        if product_id not in PRODUCT_PLAN:
            return {"success": False, "message": f"Unrecognised product: {product_id}"}

        plan, _ = PRODUCT_PLAN[product_id]
        env      = tx.get("environment", "Production")
        orig_id  = tx.get("originalTransactionId", original_transaction_id)
        exp_ms   = tx.get("expiresDate")  # milliseconds epoch or None

        expires_at = await self._activate_pro(
            user, product_id, orig_id, transaction_id, exp_ms, env, db
        )
        logger.info("[IAP] Pro activated: user=%d plan=%s env=%s", user.id, plan, env)
        return {
            "success":    True,
            "message":    f"Pro {plan} activated!",
            "expires_at": expires_at.isoformat(),
            "plan":       plan,
        }

    # ── Public: handle App Store Server Notification V2 ─────────────────────

    async def handle_notification(self, signed_payload: str, db: AsyncSession) -> bool:
        """
        Called by POST /webhook/apple.
        Verifies the outer JWS, extracts the notification, and updates Pro status.
        Always returns True (caller returns HTTP 200) unless signature verification
        fails outright — Apple retries on non-200.
        """
        # 1. Verify outer payload signature
        try:
            outer = await _verifier.verify(signed_payload, audience=Config.APNS_BUNDLE_ID)
        except Exception as e:
            logger.error("[IAP] Webhook outer JWS verification failed: %s", e)
            return False  # Return non-200 so Apple retries with a valid payload

        notification_type = outer.get("notificationType", "")
        subtype           = outer.get("subtype", "")
        data              = outer.get("data", {})

        signed_tx_info      = data.get("signedTransactionInfo")
        signed_renewal_info = data.get("signedRenewalInfo")   # noqa: F841 (reserved for future use)

        if not signed_tx_info:
            logger.warning("[IAP] Webhook has no signedTransactionInfo (type=%s)", notification_type)
            return True  # Acknowledge — nothing to act on

        # 2. Verify inner transaction JWS
        try:
            tx = await _verifier.verify(signed_tx_info)
        except Exception as e:
            logger.error("[IAP] Webhook inner transaction JWS verification failed: %s", e)
            return True  # Acknowledge to stop retries; log for investigation

        product_id              = tx.get("productId", "")
        transaction_id          = tx.get("transactionId", "")
        original_transaction_id = tx.get("originalTransactionId", "")
        env                     = tx.get("environment", "Production")
        exp_ms                  = tx.get("expiresDate")

        # 3. Resolve user from stored original_transaction_id → user mapping
        iap_tx = await db.scalar(
            select(IAPTransaction).where(
                IAPTransaction.original_transaction_id == original_transaction_id
            )
        )
        if not iap_tx:
            logger.warning(
                "[IAP] Webhook: no IAPTransaction for originalTransactionId=%s (type=%s) — "
                "user may not have called /api/iap/verify yet",
                original_transaction_id, notification_type,
            )
            return True  # Acknowledge; we'll pick this up when the user next opens the app

        user = await db.get(User, iap_tx.user_id)
        if not user:
            return True

        logger.info("[IAP] Webhook user=%d type=%s subtype=%s", user.id, notification_type, subtype)

        # 4. Apply state change
        if notification_type in GRANT_TYPES:
            await self._activate_pro(user, product_id, original_transaction_id,
                                     transaction_id, exp_ms, env, db)

        elif notification_type in GRACE_TYPES:
            # Billing failed but grace period is active — keep Pro, don't revoke
            logger.info("[IAP] Grace period / billing retry for user=%d — keeping Pro active", user.id)

        elif notification_type in REVOKE_TYPES:
            await self._revoke_pro(user, db)

        elif notification_type == "DID_CHANGE_RENEWAL_STATUS":
            # User toggled auto-renew; no immediate Pro change needed
            pass

        return True


iap_service = IAPService()
