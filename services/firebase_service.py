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
    try:
        decoded = firebase_auth.verify_id_token(id_token)
    except firebase_auth.ExpiredIdTokenError:
        raise ValueError("Firebase token has expired — ask the user to re-verify")
    except firebase_auth.InvalidIdTokenError as e:
        raise ValueError(f"Invalid Firebase token: {e}")
    except Exception as e:
        raise ValueError(f"Firebase token verification failed: {e}")

    phone = decoded.get("phone_number")
    if not phone:
        raise ValueError("Firebase token does not contain a verified phone number")

    return phone
