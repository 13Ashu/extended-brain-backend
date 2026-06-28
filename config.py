"""
Configuration Management
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""

    ENABLE_OTP = os.getenv("ENABLE_OTP", "false").lower() == "true"
    FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")  # JSON string of service account key

    # ── Founding-member program (free Pro via the FOUNDER coupon) ──────────
    # The scarcity pill ("X of N founding spots left") reads these. The pill
    # stays hidden until FOUNDING_PILL_ENABLED is flipped true (public launch);
    # until then F&F just enter the code on the upgrade screen as usual.
    FOUNDING_COUPON_CODE  = os.getenv("FOUNDING_COUPON_CODE", "FOUNDER")
    FOUNDING_SLOTS_TOTAL  = int(os.getenv("FOUNDING_SLOTS_TOTAL", "1000"))
    FOUNDING_PILL_ENABLED = os.getenv("FOUNDING_PILL_ENABLED", "false").lower() == "true"

    # MSG91 SMS — transactional OTP delivery (India). OTP generation/verification
    # stays in auth_service; MSG91 is only the SMS transport (DLT-approved template).
    MSG91_AUTH_KEY = os.getenv("MSG91_AUTH_KEY", "")
    MSG91_OTP_TEMPLATE_ID = os.getenv("MSG91_OTP_TEMPLATE_ID", "")  # DLT/MSG91 template id
    MSG91_SENDER_ID = os.getenv("MSG91_SENDER_ID", "")             # 6-char DLT header, e.g. EMINDS
    MSG91_OTP_VAR_NAME = os.getenv("MSG91_OTP_VAR_NAME", "OTP")    # variable name in the template

    # OTP anti-pumping (you pay per SMS — these cap abuse of /api/auth/send-otp)
    OTP_RESEND_COOLDOWN_SECONDS = int(os.getenv("OTP_RESEND_COOLDOWN_SECONDS", "30"))
    OTP_MAX_PER_DAY = int(os.getenv("OTP_MAX_PER_DAY", "10"))

    # WhatsApp Configuration
    WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
    WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
    WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "your_verify_token")
    WHATSAPP_API_VERSION = os.getenv("WHATSAPP_API_VERSION", "v21.0")

    # Cerebras AI
    CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL")

    # Razorpay (web dashboard payments only — not used in iOS app)
    RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
    RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
    RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

    # Apple In-App Purchase — App Store Server API
    # APPLE_ISSUER_ID:   App Store Connect → Users and Access → Integrations → Issuer ID
    # APPLE_KEY_ID:      Key ID of the App Store Connect API key (.p8)
    # APPLE_PRIVATE_KEY: Full contents of the .p8 file (-----BEGIN PRIVATE KEY----- ...)
    APPLE_ISSUER_ID = os.getenv("APPLE_ISSUER_ID", "")
    APPLE_KEY_ID = os.getenv("APPLE_KEY_ID", "")
    APPLE_PRIVATE_KEY = os.getenv("APPLE_PRIVATE_KEY", "").replace("\\n", "\n")

    # Application
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    @classmethod
    def validate_config(cls):
        """Validate that required config is present"""
        if not cls.CEREBRAS_API_KEY:
            raise ValueError("Cerebras API key not set")
