"""
Firebase phone verification service.
Verifies Firebase Auth ID tokens issued by the iOS client after SMS verification.
"""

import os
import json
import firebase_admin
from firebase_admin import auth as firebase_auth, credentials
from loguru import logger


_initialized = False


def _ensure_initialized():
    global _initialized
    if _initialized:
        return

    creds_json = os.environ.get("FIREBASE_CREDENTIALS")
    if not creds_json:
        raise RuntimeError("FIREBASE_CREDENTIALS env var is not set")

    try:
        cred_dict = json.loads(creds_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        _initialized = True
        logger.info("Firebase Admin SDK initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
        raise


def verify_phone_token(id_token: str) -> str:
    """
    Verify a Firebase phone auth ID token and return the verified E.164 phone number.
    Raises ValueError if the token is invalid or does not contain a phone_number claim.
    """
    _ensure_initialized()
    logger.info("[firebase] verify_phone_token: verifying token (len={})", len(id_token))
    try:
        decoded = firebase_auth.verify_id_token(id_token)
    except firebase_auth.ExpiredIdTokenError:
        logger.warning("[firebase] verify_phone_token: token EXPIRED")
        raise ValueError("Firebase token has expired — ask the user to re-verify")
    except firebase_auth.InvalidIdTokenError as e:
        logger.warning("[firebase] verify_phone_token: INVALID token — {}", e)
        raise ValueError(f"Invalid Firebase token: {e}")
    except Exception as e:
        logger.error("[firebase] verify_phone_token: unexpected error — {}", e)
        raise ValueError(f"Firebase token verification failed: {e}")

    phone = decoded.get("phone_number")
    if not phone:
        logger.warning("[firebase] verify_phone_token: token valid but NO phone_number claim — uid={}", decoded.get("uid") or decoded.get("sub"))
        raise ValueError("Firebase token does not contain a verified phone number")

    logger.info("[firebase] verify_phone_token: OK — phone=***{}", phone[-4:])
    return phone


def verify_google_token(id_token: str):
    """
    Verify a Firebase Google Sign-In ID token.
    Returns (uid, email, display_name) — email and name may be None.
    """
    _ensure_initialized()
    logger.info("[firebase] verify_google_token: verifying token")
    try:
        decoded = firebase_auth.verify_id_token(id_token)
    except firebase_auth.ExpiredIdTokenError:
        logger.warning("[firebase] verify_google_token: token EXPIRED")
        raise ValueError("Firebase token has expired — ask the user to sign in again")
    except firebase_auth.InvalidIdTokenError as e:
        logger.warning("[firebase] verify_google_token: INVALID token — {}", e)
        raise ValueError(f"Invalid Firebase token: {e}")
    except Exception as e:
        logger.error("[firebase] verify_google_token: unexpected error — {}", e)
        raise ValueError(f"Firebase token verification failed: {e}")

    uid = decoded.get("uid") or decoded.get("sub")
    if not uid:
        logger.warning("[firebase] verify_google_token: token valid but NO uid claim")
        raise ValueError("Firebase token does not contain a UID")

    email = decoded.get("email")
    name  = decoded.get("displayName") or decoded.get("name")
    logger.info("[firebase] verify_google_token: OK — uid={}... email={}", uid[:8], email)
    return uid, email, name


def verify_apple_token(id_token: str):
    """
    Verify a Firebase Apple Sign-In ID token.
    Returns (uid, email, display_name) — email and name may be None.
    """
    _ensure_initialized()
    logger.info("[firebase] verify_apple_token: verifying token")
    try:
        decoded = firebase_auth.verify_id_token(id_token)
    except firebase_auth.ExpiredIdTokenError:
        logger.warning("[firebase] verify_apple_token: token EXPIRED")
        raise ValueError("Firebase token has expired — ask the user to sign in again")
    except firebase_auth.InvalidIdTokenError as e:
        logger.warning("[firebase] verify_apple_token: INVALID token — {}", e)
        raise ValueError(f"Invalid Firebase token: {e}")
    except Exception as e:
        logger.error("[firebase] verify_apple_token: unexpected error — {}", e)
        raise ValueError(f"Firebase token verification failed: {e}")

    uid = decoded.get("uid") or decoded.get("sub")
    if not uid:
        logger.warning("[firebase] verify_apple_token: token valid but NO uid claim")
        raise ValueError("Firebase token does not contain a UID")

    email = decoded.get("email")
    name  = decoded.get("displayName") or decoded.get("name")
    logger.info("[firebase] verify_apple_token: OK — uid={}... email={}", uid[:8], email)
    return uid, email, name
