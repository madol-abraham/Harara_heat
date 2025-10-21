# =============================================================================
# notifications_service.py
# - Handles SMS alerts via Africa’s Talking
# - Handles push notifications via Firebase Cloud Messaging (FCM)
# =============================================================================

import os
from dotenv import load_dotenv
import africastalking
from firebase_admin import messaging

# Load environment variables
load_dotenv()

# =============================================================================
# AFRICA'S TALKING CONFIGURATION
# =============================================================================
# Make sure these are defined in your .env file:
# AT_USERNAME=sandbox
# AT_API_KEY=your_generated_api_key_here
AT_USERNAME = os.getenv("AT_USERNAME", "Madolkuol")  # default sandbox
AT_API_KEY = os.getenv("AT_API_KEY")

try:
    africastalking.initialize(AT_USERNAME, AT_API_KEY)
    sms = africastalking.SMS
    print("✅ Africa’s Talking SMS service initialized.")
except Exception as e:
    print(f" Error initializing Africa’s Talking: {e}")
    sms = None

# =============================================================================
# SMS FUNCTION
# =============================================================================
def send_sms(to_number: str, message: str):
    """
    Send an SMS using Africa’s Talking API.

    Args:
        to_number (str): Recipient phone number
        message (str): Message content

    Returns:
        dict: Status report with message ID or error
    """
    if not sms:
        return {"status": "failed", "error": "Africa's Talking not initialized"}

    try:
        response = sms.send(message, [to_number])
        print(f"✅ SMS sent to {to_number}: {response}")
        return {"status": "sent", "response": response}
    except Exception as e:
        print(f" SMS send failed: {e}")
        return {"status": "failed", "error": str(e)}

# =============================================================================
# FCM PUSH NOTIFICATION FUNCTION (optional)
# =============================================================================
def send_fcm_notification(tokens, title, body, data=None):
    """
    Send push notifications to multiple devices via Firebase Cloud Messaging.
    """
    try:
        if not tokens:
            print(" No FCM tokens provided.")
            return {"status": "no_tokens"}
        
        msg = messaging.MulticastMessage(
            notification=messaging.Notification(title=title, body=body),
            data={k: str(v) for k, v in (data or {}).items()},
            tokens=tokens
        )
        resp = messaging.send_multicast(msg)
        print(f"✅ FCM sent: {resp.success_count} success, {resp.failure_count} failure(s)")
        return {"status": "sent", "success": resp.success_count}
    except Exception as e:
        print(f" FCM error: {e}")
        return {"status": "failed", "error": str(e)}
