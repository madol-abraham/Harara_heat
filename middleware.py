# =============================================================================
# API Request Logging Middleware
# Automatically logs all incoming API requests and responses
# =============================================================================

import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from logging_service import get_logger, LogLevel, LogCategory

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Extract request info
        method = request.method
        url = str(request.url)
        endpoint = request.url.path
        user_agent = request.headers.get("user-agent", "")
        
        # Skip logging for health checks and static files
        if endpoint in ["/health", "/docs", "/openapi.json", "/redoc"] or endpoint.startswith("/static"):
            return await call_next(request)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log successful request
            get_logger().log_api_request(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                duration_ms=duration_ms,
                user_id=None,  # Could extract from auth headers if available
                error_details=None
            )
            
            return response
            
        except Exception as e:
            # Calculate duration for failed request
            duration_ms = (time.time() - start_time) * 1000
            
            # Log failed request
            get_logger().log_api_request(
                endpoint=endpoint,
                method=method,
                status_code=500,
                duration_ms=duration_ms,
                user_id=None,
                error_details=str(e)
            )
            
            # Re-raise the exception
            raise e