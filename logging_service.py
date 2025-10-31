# =============================================================================
# Harara System Logging Service
# Tracks API performance, errors, and usage patterns
# =============================================================================

import json
import datetime as dt
from typing import Optional, Dict, Any
from enum import Enum
import traceback
import time

class LogLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"

class LogCategory(str, Enum):
    API = "api"
    PREDICTION = "prediction"
    SMS = "sms"
    DATABASE = "database"
    SYSTEM = "system"
    EXPORT = "export"

class SystemLogger:
    def __init__(self, firestore_db=None):
        self.firestore_db = firestore_db
        
    def log(self, level: LogLevel, category: LogCategory, message: str, 
            details: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None,
            endpoint: Optional[str] = None, duration_ms: Optional[float] = None):
        """Log system event to Firestore"""
        try:
            log_entry = {
                "timestamp": dt.datetime.utcnow(),
                "level": level.value,
                "category": category.value,
                "message": message,
                "details": details or {},
                "user_id": user_id,
                "endpoint": endpoint,
                "duration_ms": duration_ms
            }
            
            if self.firestore_db:
                self.firestore_db.collection("system_logs").add(log_entry)
            else:
                print(f"[{level.value.upper()}] {category.value}: {message}")
                
        except Exception as e:
            print(f"Logging error: {e}")
    
    def log_api_request(self, endpoint: str, method: str, status_code: int, 
                       duration_ms: float, user_id: Optional[str] = None,
                       error_details: Optional[str] = None):
        """Log API request"""
        level = LogLevel.ERROR if status_code >= 400 else LogLevel.INFO
        message = f"{method} {endpoint} - {status_code}"
        
        details = {
            "method": method,
            "status_code": status_code,
            "response_time_ms": duration_ms
        }
        
        if error_details:
            details["error"] = error_details
            
        self.log(level, LogCategory.API, message, details, user_id, endpoint, duration_ms)
    
    def log_prediction(self, success: bool, town_count: int, duration_ms: float,
                      error_details: Optional[str] = None):
        """Log prediction operation"""
        level = LogLevel.SUCCESS if success else LogLevel.ERROR
        message = f"Prediction {'completed' if success else 'failed'} for {town_count} towns"
        
        details = {
            "town_count": town_count,
            "processing_time_ms": duration_ms
        }
        
        if error_details:
            details["error"] = error_details
            
        self.log(level, LogCategory.PREDICTION, message, details, duration_ms=duration_ms)
    
    def log_sms(self, phone: str, success: bool, provider: str, 
               error_details: Optional[str] = None):
        """Log SMS operation"""
        level = LogLevel.SUCCESS if success else LogLevel.ERROR
        message = f"SMS {'sent' if success else 'failed'} to {phone[-4:].rjust(len(phone), '*')}"
        
        details = {
            "provider": provider,
            "phone_masked": phone[-4:].rjust(len(phone), '*')
        }
        
        if error_details:
            details["error"] = error_details
            
        self.log(level, LogCategory.SMS, message, details)
    
    def log_error(self, category: LogCategory, error: Exception, context: Optional[str] = None):
        """Log system error"""
        message = f"Error in {context or 'system'}: {str(error)}"
        details = {
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "context": context
        }
        
        self.log(LogLevel.ERROR, category, message, details)

# Global logger instance
system_logger: Optional[SystemLogger] = None

def init_logger(firestore_db):
    """Initialize the global logger"""
    global system_logger
    system_logger = SystemLogger(firestore_db)
    system_logger.log(LogLevel.INFO, LogCategory.SYSTEM, "System logging initialized")

def get_logger() -> SystemLogger:
    """Get the global logger instance"""
    if system_logger is None:
        return SystemLogger()  # Fallback logger without Firestore
    return system_logger

# Decorator for automatic API logging
def log_api_call(func):
    """Decorator to automatically log API calls"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        endpoint = getattr(func, '__name__', 'unknown')
        
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            get_logger().log_api_request(endpoint, "GET", 200, duration_ms)
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            get_logger().log_api_request(endpoint, "GET", 500, duration_ms, error_details=str(e))
            raise
    return wrapper