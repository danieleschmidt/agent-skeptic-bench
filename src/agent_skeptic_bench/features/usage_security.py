"""Security and validation for usage metrics system."""

import hashlib
import hmac
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..security.input_validation import InputValidator
from ..security.rate_limiting import RateLimiter

logger = logging.getLogger(__name__)


class UsageMetricsValidator:
    """Validates usage metrics data for security and integrity."""
    
    def __init__(self):
        """Initialize usage metrics validator."""
        self.input_validator = InputValidator()
        self.rate_limiter = RateLimiter()
    
    def validate_session_creation(self, session_id: str, user_id: Optional[str] = None) -> List[str]:
        """Validate session creation parameters."""
        errors = []
        
        # Validate session ID format
        if not session_id or len(session_id) < 8:
            errors.append("Session ID must be at least 8 characters")
        
        if not session_id.replace('-', '').replace('_', '').isalnum():
            errors.append("Session ID contains invalid characters")
        
        # Validate user ID if provided
        if user_id:
            if len(user_id) > 255:
                errors.append("User ID too long (max 255 characters)")
            
            if not self.input_validator.is_safe_string(user_id):
                errors.append("User ID contains potentially dangerous characters")
        
        # Check rate limiting
        if not self.rate_limiter.check_rate_limit(f"session_creation_{user_id or 'anonymous'}", 10, 3600):
            errors.append("Too many sessions created recently (rate limit exceeded)")
        
        return errors
    
    def validate_evaluation_data(self, session_id: str, scenario_id: str, 
                                category: str, duration: float, score: float) -> List[str]:
        """Validate evaluation data parameters."""
        errors = []
        
        # Validate session ID
        if not session_id or len(session_id) < 8:
            errors.append("Invalid session ID")
        
        # Validate scenario ID
        if not scenario_id or len(scenario_id) > 255:
            errors.append("Invalid scenario ID length")
        
        if not self.input_validator.is_safe_string(scenario_id):
            errors.append("Scenario ID contains potentially dangerous characters")
        
        # Validate category
        if not category or len(category) > 100:
            errors.append("Invalid category")
        
        # Validate duration (should be reasonable)
        if duration < 0 or duration > 3600:  # Max 1 hour per evaluation
            errors.append("Duration out of valid range (0-3600 seconds)")
        
        # Validate score
        if score < 0 or score > 1:
            errors.append("Score must be between 0 and 1")
        
        return errors
    
    def validate_export_request(self, days: int, format_type: str, 
                               user_id: Optional[str] = None) -> List[str]:
        """Validate export request parameters."""
        errors = []
        
        # Validate time range
        if days < 1 or days > 365:
            errors.append("Days must be between 1 and 365")
        
        # Validate format
        if format_type not in ["json", "csv", "excel"]:
            errors.append("Invalid export format")
        
        # Check export rate limiting
        if user_id and not self.rate_limiter.check_rate_limit(f"export_{user_id}", 5, 3600):
            errors.append("Too many export requests (rate limit exceeded)")
        
        return errors
    
    def sanitize_user_input(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize user input data."""
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Remove potentially dangerous characters
                sanitized[key] = self.input_validator.sanitize_string(value)
            elif isinstance(value, (int, float)):
                # Ensure numeric values are within reasonable bounds
                if isinstance(value, float):
                    sanitized[key] = max(0, min(value, 1e6))  # Cap large values
                else:
                    sanitized[key] = max(0, min(value, 1000000))  # Cap large integers
            else:
                sanitized[key] = value
        
        return sanitized


class UsageMetricsEncryption:
    """Handles encryption of sensitive usage metrics data."""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """Initialize with optional encryption key."""
        self.encryption_enabled = encryption_key is not None
        self.key = encryption_key.encode() if encryption_key else b"default-demo-key"
    
    def encrypt_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive fields in usage data."""
        if not self.encryption_enabled:
            return data
        
        encrypted_data = data.copy()
        
        # Fields that should be encrypted
        sensitive_fields = ["user_id", "api_calls", "scenarios_completed"]
        
        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field]:
                try:
                    # Simple HMAC-based encryption for demo
                    original_value = str(encrypted_data[field])
                    encrypted_value = hmac.new(
                        self.key, 
                        original_value.encode(), 
                        hashlib.sha256
                    ).hexdigest()
                    encrypted_data[f"{field}_encrypted"] = encrypted_value
                    
                    # Remove original for security
                    del encrypted_data[field]
                    
                except Exception as e:
                    logger.warning(f"Failed to encrypt field {field}: {e}")
        
        return encrypted_data
    
    def hash_identifier(self, identifier: str) -> str:
        """Create a hash of an identifier for privacy."""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]


class UsageDataRetentionManager:
    """Manages data retention policies for usage metrics."""
    
    def __init__(self, retention_days: int = 90):
        """Initialize with retention policy."""
        self.retention_days = retention_days
    
    def cleanup_old_data(self, storage_path: Path) -> Dict[str, Any]:
        """Clean up old usage metrics data based on retention policy."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        cleanup_stats = {
            "files_checked": 0,
            "files_deleted": 0,
            "records_purged": 0,
            "errors": []
        }
        
        try:
            for file_path in storage_path.glob("usage_metrics_*.jsonl"):
                cleanup_stats["files_checked"] += 1
                
                try:
                    # Parse date from filename
                    date_str = file_path.stem.split("_")[-1]  # Get date part
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    if file_date < cutoff_date:
                        # File is older than retention period
                        lines_before = sum(1 for _ in open(file_path))
                        file_path.unlink()  # Delete file
                        
                        cleanup_stats["files_deleted"] += 1
                        cleanup_stats["records_purged"] += lines_before
                        logger.info(f"Deleted old usage metrics file: {file_path}")
                
                except Exception as e:
                    error_msg = f"Failed to process file {file_path}: {e}"
                    cleanup_stats["errors"].append(error_msg)
                    logger.error(error_msg)
        
        except Exception as e:
            error_msg = f"Failed to cleanup usage data: {e}"
            cleanup_stats["errors"].append(error_msg)
            logger.error(error_msg)
        
        return cleanup_stats
    
    def archive_old_data(self, storage_path: Path, archive_path: Path) -> Dict[str, Any]:
        """Archive old usage metrics data instead of deleting."""
        archive_path.mkdir(parents=True, exist_ok=True)
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days // 2)  # Archive at halfway point
        
        archive_stats = {
            "files_checked": 0,
            "files_archived": 0,
            "records_archived": 0,
            "errors": []
        }
        
        try:
            for file_path in storage_path.glob("usage_metrics_*.jsonl"):
                archive_stats["files_checked"] += 1
                
                try:
                    # Parse date from filename
                    date_str = file_path.stem.split("_")[-1]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    if file_date < cutoff_date:
                        # Move to archive
                        archive_file_path = archive_path / file_path.name
                        lines_count = sum(1 for _ in open(file_path))
                        
                        file_path.rename(archive_file_path)
                        
                        archive_stats["files_archived"] += 1
                        archive_stats["records_archived"] += lines_count
                        logger.info(f"Archived usage metrics file: {file_path} -> {archive_file_path}")
                
                except Exception as e:
                    error_msg = f"Failed to archive file {file_path}: {e}"
                    archive_stats["errors"].append(error_msg)
                    logger.error(error_msg)
        
        except Exception as e:
            error_msg = f"Failed to archive usage data: {e}"
            archive_stats["errors"].append(error_msg)
            logger.error(error_msg)
        
        return archive_stats


class UsageAnomalyDetector:
    """Detects anomalies in usage patterns for security monitoring."""
    
    def __init__(self):
        """Initialize anomaly detector."""
        self.baseline_metrics = {}
    
    def analyze_session_patterns(self, session_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze session patterns for anomalies."""
        if not session_metrics:
            return {"anomalies": [], "alerts": []}
        
        anomalies = []
        alerts = []
        
        # Check for unusual evaluation volumes
        eval_counts = [m.get("evaluation_count", 0) for m in session_metrics]
        if eval_counts:
            avg_evals = sum(eval_counts) / len(eval_counts)
            max_evals = max(eval_counts)
            
            # Alert if any session has >3x average evaluations
            if max_evals > avg_evals * 3 and max_evals > 10:
                anomalies.append({
                    "type": "high_evaluation_volume",
                    "description": f"Session with {max_evals} evaluations (avg: {avg_evals:.1f})",
                    "severity": "medium"
                })
        
        # Check for unusual duration patterns
        durations = [m.get("total_duration", 0) for m in session_metrics]
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            
            # Alert if any session >5x average duration
            if max_duration > avg_duration * 5 and max_duration > 300:  # >5 minutes
                anomalies.append({
                    "type": "long_session_duration",
                    "description": f"Session lasted {max_duration:.1f}s (avg: {avg_duration:.1f}s)",
                    "severity": "low"
                })
        
        # Check for rapid session creation (potential abuse)
        timestamps = [datetime.fromisoformat(m.get("timestamp", datetime.utcnow().isoformat())) 
                     for m in session_metrics]
        timestamps.sort()
        
        rapid_sessions = 0
        for i in range(1, len(timestamps)):
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            if time_diff < 10:  # Less than 10 seconds between sessions
                rapid_sessions += 1
        
        if rapid_sessions > 3:
            alerts.append({
                "type": "rapid_session_creation",
                "description": f"{rapid_sessions} sessions created within 10 seconds of each other",
                "severity": "high"
            })
        
        return {
            "anomalies": anomalies,
            "alerts": alerts,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "sessions_analyzed": len(session_metrics)
        }
    
    def check_export_patterns(self, export_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for suspicious export patterns."""
        if not export_requests:
            return {"alerts": []}
        
        alerts = []
        
        # Check for excessive export requests
        recent_exports = [
            req for req in export_requests 
            if datetime.fromisoformat(req.get("timestamp", datetime.utcnow().isoformat())) 
            > datetime.utcnow() - timedelta(hours=1)
        ]
        
        if len(recent_exports) > 10:
            alerts.append({
                "type": "excessive_exports",
                "description": f"{len(recent_exports)} export requests in the last hour",
                "severity": "medium"
            })
        
        # Check for large data exports
        large_exports = [
            req for req in export_requests
            if req.get("records_exported", 0) > 10000
        ]
        
        if large_exports:
            alerts.append({
                "type": "large_data_export", 
                "description": f"{len(large_exports)} exports with >10K records",
                "severity": "low"
            })
        
        return {
            "alerts": alerts,
            "total_exports": len(export_requests),
            "recent_exports": len(recent_exports)
        }