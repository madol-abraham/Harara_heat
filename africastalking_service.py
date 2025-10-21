# africastalking_service.py
import africastalking
import os
from dotenv import load_dotenv

load_dotenv()

# Load credentials
AT_USERNAME = os.getenv("AT_USERNAME", "Madolkuol")  # default sandbox
AT_API_KEY = os.getenv("AT_API_KEY")

# Initialize SDK
africastalking.initialize(AT_USERNAME, AT_API_KEY)

sms = africastalking.SMS

def send_sms_africa(phone_number: str, message: str):
    """Send SMS using Africa's Talking"""
    try:
        response = sms.send(message, [phone_number])
        print(f"✅ SMS sent to {phone_number}: {response}")
        return {"status": "sent", "response": response}
    except Exception as e:
        print(f" SMS failed for {phone_number}: {e}")
        return {"status": "failed", "error": str(e)}
