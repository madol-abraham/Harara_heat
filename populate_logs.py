# =============================================================================
# Populate Initial System Logs
# Creates sample system logs for testing the export functionality
# =============================================================================

import os
import json
import datetime as dt
from zoneinfo import ZoneInfo
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

def init_firestore():
    """Initialize Firebase Firestore"""
    try:
        firebase_key = os.getenv("FIREBASE_SERVICE_KEY")
        if firebase_key:
            firebase_info = json.loads(firebase_key)
            cred = credentials.Certificate(firebase_info)
            firebase_admin.initialize_app(cred)
            return firestore.client()
        else:
            cred_path = "firebase-key.json"
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            return firestore.client()
    except Exception as e:
        print(f"❌ Firestore init error: {e}")
        return None

def populate_sample_logs():
    """Create sample system logs"""
    db = init_firestore()
    if not db:
        print("Failed to initialize Firestore")
        return
    
    # Sample log entries
    sample_logs = [
        {
            "timestamp": dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2),
            "level": "info",
            "category": "system",
            "message": "Harara API started successfully",
            "details": {"version": "2.0.1"},
            "endpoint": None,
            "duration_ms": None
        },
        {
            "timestamp": dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1, minutes=30),
            "level": "success",
            "category": "prediction",
            "message": "Prediction completed for 6 towns",
            "details": {"town_count": 6, "processing_time_ms": 15420},
            "endpoint": "/predict/run",
            "duration_ms": 15420
        },
        {
            "timestamp": dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1),
            "level": "info",
            "category": "api",
            "message": "GET /firestore/predictions/today - 200",
            "details": {"method": "GET", "status_code": 200, "response_time_ms": 245},
            "endpoint": "/firestore/predictions/today",
            "duration_ms": 245
        },
        {
            "timestamp": dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=45),
            "level": "success",
            "category": "sms",
            "message": "SMS sent to ****3010",
            "details": {"provider": "africastalking", "phone_masked": "****3010"},
            "endpoint": None,
            "duration_ms": None
        },
        {
            "timestamp": dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=30),
            "level": "info",
            "category": "api",
            "message": "POST /alerts/manual - 200",
            "details": {"method": "POST", "status_code": 200, "response_time_ms": 1250},
            "endpoint": "/alerts/manual",
            "duration_ms": 1250
        },
        {
            "timestamp": dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=15),
            "level": "warning",
            "category": "system",
            "message": "High API response time detected",
            "details": {"avg_response_time_ms": 2500, "threshold_ms": 2000},
            "endpoint": None,
            "duration_ms": None
        },
        {
            "timestamp": dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5),
            "level": "success",
            "category": "database",
            "message": "User registered successfully for Juba",
            "details": {"town": "Juba"},
            "endpoint": "/users/register",
            "duration_ms": None
        }
    ]
    
    # Add logs to Firestore
    for log_entry in sample_logs:
        try:
            db.collection("system_logs").add(log_entry)
            print(f"Added log: {log_entry['message']}")
        except Exception as e:
            print(f"Failed to add log: {e}")
    
    print(f"\nSuccessfully populated {len(sample_logs)} sample system logs!")
    print("You can now test the /api/export/logs endpoint")

if __name__ == "__main__":
    populate_sample_logs()