"""
SMS Service — MSG91 transactional SMS for OTP delivery (India).

OTP generation, storage, expiry and verification all stay in `auth_service` +
the `otp_verifications` table. This module is purely the SMS *transport*: it
hands a code to MSG91's Flow API (v5), which renders it into a DLT-approved
template and sends the SMS. It is deliberately separate from any messaging channel.

Required env (see config.py):
    MSG91_AUTH_KEY          MSG91 account auth key
    MSG91_OTP_TEMPLATE_ID   DLT/MSG91 template id for the OTP message
    MSG91_SENDER_ID         6-char DLT header (e.g. EMINDS)   [optional for Flow]
    MSG91_OTP_VAR_NAME      variable name in the template (default "OTP")

The registered DLT template must contain a single variable, e.g.:
    ##OTP## is your OTP for Extended Minds login. Valid for 10 minutes. Do not
    share this code with anyone.
where MSG91_OTP_VAR_NAME == "OTP".
"""

import httpx
from loguru import logger

from config import Config


class SMSService:
    """Sends OTP SMS via MSG91's Flow API. Stateless; safe to use as a singleton."""

    FLOW_URL = "https://control.msg91.com/api/v5/flow"
    TIMEOUT_SECONDS = 15.0

    @property
    def is_configured(self) -> bool:
        return bool(Config.MSG91_AUTH_KEY and Config.MSG91_OTP_TEMPLATE_ID)

    @staticmethod
    def _normalize(phone_number: str) -> str:
        """MSG91 wants the number with country code and no '+', spaces or dashes."""
        return (
            phone_number.strip()
            .lstrip("+")
            .replace(" ", "")
            .replace("-", "")
        )

    async def send_otp(self, phone_number: str, otp_code: str) -> dict:
        """
        Send `otp_code` to `phone_number` via MSG91. Raises on failure so the
        caller can roll back the stored OTP and surface an error to the client.
        """
        if not self.is_configured:
            logger.error(
                "[sms] MSG91 not configured — set MSG91_AUTH_KEY and "
                "MSG91_OTP_TEMPLATE_ID"
            )
            raise RuntimeError("SMS provider not configured")

        recipient = {
            "mobiles": self._normalize(phone_number),
            Config.MSG91_OTP_VAR_NAME: otp_code,
        }
        payload = {
            "template_id": Config.MSG91_OTP_TEMPLATE_ID,
            "recipients": [recipient],
        }
        if Config.MSG91_SENDER_ID:
            payload["sender"] = Config.MSG91_SENDER_ID

        headers = {
            "authkey": Config.MSG91_AUTH_KEY,
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT_SECONDS) as client:
                resp = await client.post(self.FLOW_URL, json=payload, headers=headers)
        except httpx.HTTPError as e:
            logger.error("[sms] MSG91 request error: {}", e)
            raise RuntimeError("SMS send failed (network)") from e

        body: dict = {}
        try:
            body = resp.json()
        except Exception:
            pass

        # MSG91 returns HTTP 200 with {"type": "success", ...} on success and
        # {"type": "error", "message": ...} (or a non-2xx) on failure.
        if resp.status_code != 200 or (isinstance(body, dict) and body.get("type") == "error"):
            detail = body or resp.text
            logger.error("[sms] MSG91 send failed status={} body={}", resp.status_code, detail)
            raise RuntimeError(f"SMS send failed: {detail}")

        logger.info(
            "[sms] OTP SMS sent to ***{} request_id={}",
            phone_number[-4:],
            body.get("request_id") if isinstance(body, dict) else None,
        )
        return body


# Module-level singleton — import and call `sms_service.send_otp(...)`.
sms_service = SMSService()
