# Harara API System Logging Implementation

## Overview
Implemented comprehensive system logging for the Harara Heatwave API to track performance, errors, and usage patterns. All logs are stored in Firestore and can be exported via the `/api/export/logs` endpoint.

## Components Implemented

### 1. Logging Service (`logging_service.py`)
- **SystemLogger class**: Centralized logging with Firestore integration
- **Log levels**: INFO, WARNING, ERROR, SUCCESS
- **Log categories**: API, PREDICTION, SMS, DATABASE, SYSTEM, EXPORT
- **Specialized methods**:
  - `log_api_request()`: Track API calls with timing
  - `log_prediction()`: Track ML prediction operations
  - `log_sms()`: Track SMS sending operations
  - `log_error()`: Track system errors with stack traces

### 2. Request Logging Middleware (`middleware.py`)
- **Automatic API logging**: Captures all HTTP requests/responses
- **Performance tracking**: Records response times
- **Error tracking**: Logs failed requests with details
- **Selective logging**: Skips health checks and static files

### 3. Integration Points
- **Main API**: Integrated logging throughout `main.py`
- **Startup logging**: System initialization events
- **Prediction logging**: ML model execution tracking
- **SMS logging**: Message delivery tracking
- **Database logging**: User registration and data operations
- **Scheduled job logging**: Daily prediction runs

### 4. Sample Data Population (`populate_logs.py`)
- Creates initial system logs for testing
- Demonstrates various log types and categories
- Enables immediate testing of export functionality

## Log Structure
Each log entry contains:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "info|warning|error|success",
  "category": "api|prediction|sms|database|system|export",
  "message": "Human-readable description",
  "details": {
    "additional_context": "value",
    "performance_metrics": 1250
  },
  "endpoint": "/api/endpoint",
  "duration_ms": 1250,
  "user_id": "optional_user_id"
}
```

## Tracked Events

### API Operations
- All HTTP requests with method, endpoint, status code
- Response times and error details
- User identification (when available)

### Prediction Operations
- ML model execution start/completion
- Processing times and town counts
- Success/failure status with error details

### SMS Operations
- Message sending attempts
- Provider used (Africa's Talking/simulation)
- Success/failure status
- Phone number masking for privacy

### Database Operations
- User registrations
- Data retrieval operations
- Connection issues and errors

### System Operations
- API startup/shutdown
- Scheduled job execution
- Performance warnings
- Configuration changes

## Export Functionality
The `/api/export/logs` endpoint now works properly:
- Exports all system logs as CSV
- Includes all log fields and metadata
- Timestamped filename for organization
- Handles empty collections gracefully

## Usage Examples

### Manual Logging
```python
from logging_service import get_logger, LogLevel, LogCategory

# Log an info message
get_logger().log(LogLevel.INFO, LogCategory.API, "User action completed")

# Log with details
get_logger().log(LogLevel.SUCCESS, LogCategory.PREDICTION, 
                "Prediction completed", {"towns": 6, "duration_ms": 1500})

# Log an error
get_logger().log_error(LogCategory.DATABASE, exception, "user_registration")
```

### Automatic Logging
- All API requests are automatically logged via middleware
- Prediction operations log start/completion automatically
- SMS operations log success/failure automatically
- Database operations log key events automatically

## Benefits

### Operational Monitoring
- Track API performance and identify bottlenecks
- Monitor prediction accuracy and processing times
- Track SMS delivery success rates
- Identify system errors and patterns

### Debugging & Troubleshooting
- Detailed error logs with stack traces
- Request/response correlation
- Performance metrics for optimization
- Historical data for trend analysis

### Compliance & Auditing
- Complete audit trail of system operations
- User action tracking (when authenticated)
- Data export capabilities
- Structured logging for analysis

## Testing
1. **Populated sample logs**: 7 different log types created
2. **Export endpoint**: `/api/export/logs` now returns CSV data
3. **Real-time logging**: All API operations now generate logs
4. **Error handling**: Failed operations are properly logged

## Next Steps
1. **Dashboard integration**: Display logs in web dashboard
2. **Log rotation**: Implement automatic cleanup of old logs
3. **Alerting**: Set up notifications for critical errors
4. **Analytics**: Create dashboards for log analysis
5. **Performance optimization**: Index frequently queried fields

## Files Modified/Created
- ✅ `logging_service.py` - Core logging functionality
- ✅ `middleware.py` - Automatic request logging
- ✅ `main.py` - Integrated logging throughout API
- ✅ `populate_logs.py` - Sample data creation
- ✅ `export_routes.py` - Already supported system logs export

The system logging implementation is now complete and fully functional. The `/api/export/logs` endpoint will return proper CSV data with all system logs, enabling comprehensive monitoring and analysis of the Harara API operations.