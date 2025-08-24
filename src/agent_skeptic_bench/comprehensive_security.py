"""Comprehensive Security Framework for Agent Skeptic Bench

Provides input validation, rate limiting, authentication, authorization,
audit logging, and security monitoring for production deployment.
"""

import hashlib
import hmac
import json
import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Union
import secrets
from functools import wraps

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Security event types."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INPUT_VALIDATION = "input_validation"
    RATE_LIMIT = "rate_limit"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    SYSTEM_ACCESS = "system_access"


@dataclass
class SecurityEvent:
    """Security event record."""
    event_type: EventType
    level: SecurityLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitRule:
    """Rate limiting rule definition."""
    name: str
    max_requests: int
    time_window_seconds: int
    scope: str  # 'ip', 'user', 'session', 'global'
    block_duration_seconds: int = 300


class InputValidator:
    """Comprehensive input validation."""
    
    # Common dangerous patterns
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # XSS scripts
        r'javascript:',  # JavaScript URLs
        r'data:text/html',  # Data URLs with HTML
        r'vbscript:',  # VBScript URLs
        r'on\w+\s*=',  # Event handlers
        r'expression\s*\(',  # CSS expressions
        r'<!--.*?-->',  # HTML comments (potential XSS)
        r'union.*select',  # SQL injection
        r'drop\s+table',  # SQL injection
        r'insert\s+into',  # SQL injection
        r'\|\|\s*sleep',  # NoSQL injection
        r'\$\{.*\}',  # Template injection
        r'<%.*%>',  # Server-side template injection
    ]
    
    # File type restrictions
    ALLOWED_FILE_EXTENSIONS = {'.txt', '.json', '.csv', '.md', '.log'}
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    def __init__(self):
        """Initialize input validator."""
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS]
        self.validation_cache: Dict[str, bool] = {}
        self.cache_max_size = 10000
    
    def validate_string_input(self, value: str, max_length: int = 10000,
                            allow_html: bool = False) -> Dict[str, Any]:
        """Validate string input for security issues."""
        validation_result = {
            "valid": True,
            "issues": [],
            "sanitized_value": value,
            "risk_level": SecurityLevel.LOW
        }
        
        # Check cache first
        cache_key = hashlib.md5(f"{value}:{max_length}:{allow_html}".encode()).hexdigest()
        if cache_key in self.validation_cache:
            return {"valid": self.validation_cache[cache_key], "cached": True}
        
        # Length validation
        if len(value) > max_length:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Input exceeds maximum length: {len(value)} > {max_length}")
            validation_result["risk_level"] = SecurityLevel.MEDIUM
        
        # Pattern matching for dangerous content
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(value):
                validation_result["valid"] = False
                validation_result["issues"].append(f"Dangerous pattern detected: {self.DANGEROUS_PATTERNS[i]}")
                validation_result["risk_level"] = SecurityLevel.HIGH
        
        # HTML validation if not allowed
        if not allow_html and ('<' in value or '>' in value):
            validation_result["valid"] = False
            validation_result["issues"].append("HTML content not allowed")
            validation_result["risk_level"] = SecurityLevel.MEDIUM
        
        # Unicode normalization attacks
        normalized = value.encode('utf-8', 'ignore').decode('utf-8')
        if normalized != value:
            validation_result["sanitized_value"] = normalized
            validation_result["issues"].append("Input contained invalid Unicode sequences")
        
        # Update cache
        if len(self.validation_cache) < self.cache_max_size:
            self.validation_cache[cache_key] = validation_result["valid"]
        
        return validation_result
    
    def validate_json_input(self, json_str: str, max_depth: int = 10,
                          max_keys: int = 1000) -> Dict[str, Any]:
        """Validate JSON input for security and structure."""
        validation_result = {
            "valid": True,
            "issues": [],
            "parsed_data": None,
            "risk_level": SecurityLevel.LOW
        }
        
        try:
            # Parse JSON
            data = json.loads(json_str)
            validation_result["parsed_data"] = data
            
            # Check depth
            def check_depth(obj, current_depth=0):
                if current_depth > max_depth:
                    return False
                if isinstance(obj, dict):
                    return all(check_depth(v, current_depth + 1) for v in obj.values())
                elif isinstance(obj, list):
                    return all(check_depth(item, current_depth + 1) for item in obj)
                return True
            
            if not check_depth(data):
                validation_result["valid"] = False
                validation_result["issues"].append(f"JSON depth exceeds maximum: {max_depth}")
                validation_result["risk_level"] = SecurityLevel.HIGH
            
            # Count total keys (recursive)
            def count_keys(obj):
                if isinstance(obj, dict):
                    return len(obj) + sum(count_keys(v) for v in obj.values())
                elif isinstance(obj, list):
                    return sum(count_keys(item) for item in obj)
                return 0
            
            total_keys = count_keys(data)
            if total_keys > max_keys:
                validation_result["valid"] = False
                validation_result["issues"].append(f"JSON key count exceeds maximum: {total_keys} > {max_keys}")
                validation_result["risk_level"] = SecurityLevel.MEDIUM
            
        except json.JSONDecodeError as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"Invalid JSON format: {str(e)}")
            validation_result["risk_level"] = SecurityLevel.MEDIUM
        
        return validation_result
    
    def validate_filename(self, filename: str) -> Dict[str, Any]:
        """Validate filename for security issues."""
        validation_result = {
            "valid": True,
            "issues": [],
            "sanitized_filename": filename,
            "risk_level": SecurityLevel.LOW
        }
        
        # Path traversal check
        if '..' in filename or '/' in filename or '\\' in filename:
            validation_result["valid"] = False
            validation_result["issues"].append("Path traversal attempt detected")
            validation_result["risk_level"] = SecurityLevel.HIGH
        
        # File extension check
        if '.' in filename:
            ext = '.' + filename.split('.')[-1].lower()
            if ext not in self.ALLOWED_FILE_EXTENSIONS:
                validation_result["valid"] = False
                validation_result["issues"].append(f"File extension not allowed: {ext}")
                validation_result["risk_level"] = SecurityLevel.MEDIUM
        
        # Length check
        if len(filename) > 255:
            validation_result["valid"] = False
            validation_result["issues"].append("Filename too long")
            validation_result["risk_level"] = SecurityLevel.MEDIUM
        
        # Control character check
        if any(ord(c) < 32 for c in filename):
            validation_result["valid"] = False
            validation_result["issues"].append("Control characters in filename")
            validation_result["risk_level"] = SecurityLevel.HIGH
        
        return validation_result


class RateLimiter:
    """Advanced rate limiting with multiple scopes and rules."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self.rules: Dict[str, RateLimitRule] = {}
        self.request_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self.blocked_entities: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = datetime.utcnow()
    
    def add_rule(self, rule: RateLimitRule):
        """Add a rate limiting rule."""
        self.rules[rule.name] = rule
        logger.info(f"Added rate limit rule: {rule.name} - {rule.max_requests}/{rule.time_window_seconds}s")
    
    def check_rate_limit(self, rule_name: str, identifier: str,
                        source_ip: Optional[str] = None) -> Dict[str, Any]:
        """Check if request is within rate limits."""
        if rule_name not in self.rules:
            return {"allowed": True, "error": f"Rule {rule_name} not found"}
        
        rule = self.rules[rule_name]
        now = datetime.utcnow()
        
        # Determine scope key
        if rule.scope == 'ip':
            scope_key = source_ip or identifier
        elif rule.scope == 'user':
            scope_key = identifier
        elif rule.scope == 'session':
            scope_key = identifier
        else:  # global
            scope_key = 'global'
        
        # Check if currently blocked
        if scope_key in self.blocked_entities[rule_name]:
            block_expires = self.blocked_entities[rule_name][scope_key]
            if now < block_expires:
                remaining = (block_expires - now).total_seconds()
                return {
                    "allowed": False,
                    "reason": "rate_limit_exceeded",
                    "retry_after": remaining,
                    "rule": rule_name
                }
            else:
                # Block expired, remove it
                del self.blocked_entities[rule_name][scope_key]
        
        # Clean old requests
        cutoff = now - timedelta(seconds=rule.time_window_seconds)
        history = self.request_history[rule_name][scope_key]
        while history and history[0] < cutoff:
            history.popleft()
        
        # Check current request count
        if len(history) >= rule.max_requests:
            # Rate limit exceeded - block entity
            block_until = now + timedelta(seconds=rule.block_duration_seconds)
            self.blocked_entities[rule_name][scope_key] = block_until
            
            return {
                "allowed": False,
                "reason": "rate_limit_exceeded",
                "retry_after": rule.block_duration_seconds,
                "rule": rule_name,
                "requests_count": len(history),
                "max_requests": rule.max_requests
            }
        
        # Add current request to history
        history.append(now)
        
        # Periodic cleanup
        if now - self.last_cleanup > timedelta(seconds=self.cleanup_interval):
            self._cleanup_old_data()
        
        return {
            "allowed": True,
            "requests_count": len(history),
            "max_requests": rule.max_requests,
            "time_window": rule.time_window_seconds
        }
    
    def _cleanup_old_data(self):
        """Clean up old rate limiting data."""
        now = datetime.utcnow()
        
        # Clean expired blocks
        for rule_name in self.blocked_entities:
            expired_keys = [
                key for key, expires in self.blocked_entities[rule_name].items()
                if now >= expires
            ]
            for key in expired_keys:
                del self.blocked_entities[rule_name][key]
        
        # Clean old request history
        for rule_name, rule in self.rules.items():
            cutoff = now - timedelta(seconds=rule.time_window_seconds * 2)  # Keep extra history
            for scope_key in list(self.request_history[rule_name].keys()):
                history = self.request_history[rule_name][scope_key]
                while history and history[0] < cutoff:
                    history.popleft()
                
                # Remove empty histories
                if not history:
                    del self.request_history[rule_name][scope_key]
        
        self.last_cleanup = now
        logger.debug("Completed rate limiter cleanup")
    
    def get_rate_limit_status(self, rule_name: str, identifier: str) -> Dict[str, Any]:
        """Get current rate limit status for an identifier."""
        if rule_name not in self.rules:
            return {"error": f"Rule {rule_name} not found"}
        
        rule = self.rules[rule_name]
        now = datetime.utcnow()
        
        # Determine scope key
        if rule.scope == 'user':
            scope_key = identifier
        else:
            scope_key = identifier
        
        # Check if blocked
        if scope_key in self.blocked_entities[rule_name]:
            block_expires = self.blocked_entities[rule_name][scope_key]
            if now < block_expires:
                return {
                    "blocked": True,
                    "block_expires": block_expires.isoformat(),
                    "retry_after": (block_expires - now).total_seconds()
                }
        
        # Get current usage
        cutoff = now - timedelta(seconds=rule.time_window_seconds)
        history = self.request_history[rule_name][scope_key]
        current_count = sum(1 for req_time in history if req_time > cutoff)
        
        return {
            "blocked": False,
            "current_requests": current_count,
            "max_requests": rule.max_requests,
            "time_window": rule.time_window_seconds,
            "requests_remaining": rule.max_requests - current_count
        }


class SecurityAuditLogger:
    """Comprehensive security audit logging."""
    
    def __init__(self, max_events: int = 100000):
        """Initialize security audit logger."""
        self.events: deque = deque(maxlen=max_events)
        self.event_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.suspicious_ips: Set[str] = set()
        self.last_analysis = datetime.utcnow()
    
    def log_event(self, event_type: EventType, level: SecurityLevel,
                 message: str, source_ip: Optional[str] = None,
                 user_id: Optional[str] = None, session_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """Log a security event."""
        event = SecurityEvent(
            event_type=event_type,
            level=level,
            message=message,
            source_ip=source_ip,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {}
        )
        
        self.events.append(event)
        
        # Update counts
        self.event_counts[event_type.value][level.value] += 1
        
        # Log to system logger
        log_level = {
            SecurityLevel.LOW: logging.INFO,
            SecurityLevel.MEDIUM: logging.WARNING,
            SecurityLevel.HIGH: logging.ERROR,
            SecurityLevel.CRITICAL: logging.CRITICAL
        }.get(level, logging.INFO)
        
        logger.log(log_level, f"SECURITY [{level.value.upper()}] {event_type.value}: {message}")
        
        # Analyze for suspicious activity
        if level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            self._analyze_suspicious_activity(event)
    
    def _analyze_suspicious_activity(self, event: SecurityEvent):
        """Analyze event for suspicious patterns."""
        if event.source_ip:
            # Count recent high-severity events from this IP
            recent_events = [
                e for e in self.events
                if e.source_ip == event.source_ip
                and e.level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]
                and (datetime.utcnow() - e.timestamp).total_seconds() < 3600  # Last hour
            ]
            
            if len(recent_events) >= 3:
                self.suspicious_ips.add(event.source_ip)
                logger.critical(f"SUSPICIOUS IP DETECTED: {event.source_ip} - {len(recent_events)} high-severity events")
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for the specified time period."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_events = [e for e in self.events if e.timestamp > cutoff]
        
        # Count by type and level
        summary = {
            "total_events": len(recent_events),
            "by_type": defaultdict(int),
            "by_level": defaultdict(int),
            "by_ip": defaultdict(int),
            "suspicious_ips": list(self.suspicious_ips),
            "time_period_hours": hours
        }
        
        for event in recent_events:
            summary["by_type"][event.event_type.value] += 1
            summary["by_level"][event.level.value] += 1
            if event.source_ip:
                summary["by_ip"][event.source_ip] += 1
        
        # Top IPs by event count
        top_ips = sorted(summary["by_ip"].items(), key=lambda x: x[1], reverse=True)[:10]
        summary["top_active_ips"] = [{"ip": ip, "events": count} for ip, count in top_ips]
        
        return dict(summary)
    
    def get_events(self, event_type: Optional[EventType] = None,
                  level: Optional[SecurityLevel] = None,
                  hours: int = 24,
                  source_ip: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get filtered security events."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        filtered_events = []
        
        for event in self.events:
            if event.timestamp < cutoff:
                continue
            
            if event_type and event.event_type != event_type:
                continue
            
            if level and event.level != level:
                continue
            
            if source_ip and event.source_ip != source_ip:
                continue
            
            filtered_events.append({
                "event_type": event.event_type.value,
                "level": event.level.value,
                "message": event.message,
                "timestamp": event.timestamp.isoformat(),
                "source_ip": event.source_ip,
                "user_id": event.user_id,
                "session_id": event.session_id,
                "metadata": event.metadata
            })
        
        return filtered_events


class ComprehensiveSecurity:
    """Main security manager coordinating all security components."""
    
    def __init__(self):
        """Initialize comprehensive security system."""
        self.input_validator = InputValidator()
        self.rate_limiter = RateLimiter()
        self.audit_logger = SecurityAuditLogger()
        
        # Security configuration
        self.security_enabled = True
        self.strict_mode = False
        
        self._setup_default_rate_limits()
    
    def _setup_default_rate_limits(self):
        """Set up default rate limiting rules."""
        default_rules = [
            RateLimitRule("api_requests", 100, 60, "ip"),  # 100 req/min per IP
            RateLimitRule("evaluation_requests", 10, 60, "user"),  # 10 evaluations/min per user
            RateLimitRule("auth_attempts", 5, 300, "ip", 900),  # 5 auth attempts per 5min, block for 15min
            RateLimitRule("global_load", 1000, 60, "global")  # 1000 req/min globally
        ]
        
        for rule in default_rules:
            self.rate_limiter.add_rule(rule)
    
    def validate_and_sanitize_input(self, data: Any, context: str = "general") -> Dict[str, Any]:
        """Comprehensive input validation and sanitization."""
        validation_result = {
            "valid": True,
            "sanitized_data": data,
            "issues": [],
            "risk_level": SecurityLevel.LOW
        }
        
        try:
            if isinstance(data, str):
                # String validation
                string_result = self.input_validator.validate_string_input(data)
                validation_result.update(string_result)
                
            elif isinstance(data, dict):
                # Dictionary validation - check each value
                sanitized_dict = {}
                for key, value in data.items():
                    key_result = self.validate_and_sanitize_input(str(key), f"{context}.key")
                    value_result = self.validate_and_sanitize_input(value, f"{context}.{key}")
                    
                    if not key_result["valid"] or not value_result["valid"]:
                        validation_result["valid"] = False
                        validation_result["issues"].extend(key_result.get("issues", []))
                        validation_result["issues"].extend(value_result.get("issues", []))
                    
                    sanitized_dict[key_result.get("sanitized_data", key)] = value_result.get("sanitized_data", value)
                
                validation_result["sanitized_data"] = sanitized_dict
                
            elif isinstance(data, list):
                # List validation - check each item
                sanitized_list = []
                for i, item in enumerate(data):
                    item_result = self.validate_and_sanitize_input(item, f"{context}[{i}]")
                    if not item_result["valid"]:
                        validation_result["valid"] = False
                        validation_result["issues"].extend(item_result.get("issues", []))
                    sanitized_list.append(item_result.get("sanitized_data", item))
                
                validation_result["sanitized_data"] = sanitized_list
            
            # Log validation results if issues found
            if not validation_result["valid"]:
                self.audit_logger.log_event(
                    EventType.INPUT_VALIDATION,
                    validation_result["risk_level"],
                    f"Input validation failed in {context}: {'; '.join(validation_result['issues'])}",
                    metadata={"context": context, "issues": validation_result["issues"]}
                )
        
        except Exception as e:
            validation_result = {
                "valid": False,
                "sanitized_data": None,
                "issues": [f"Validation error: {str(e)}"],
                "risk_level": SecurityLevel.HIGH
            }
            
            self.audit_logger.log_event(
                EventType.INPUT_VALIDATION,
                SecurityLevel.HIGH,
                f"Input validation exception in {context}: {str(e)}",
                metadata={"context": context, "exception": str(e)}
            )
        
        return validation_result
    
    def check_request_security(self, request_data: Dict[str, Any],
                             source_ip: Optional[str] = None,
                             user_id: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive security check for incoming requests."""
        security_result = {
            "allowed": True,
            "issues": [],
            "sanitized_data": request_data,
            "security_level": SecurityLevel.LOW
        }
        
        # Input validation
        validation_result = self.validate_and_sanitize_input(request_data, "request")
        if not validation_result["valid"]:
            security_result["allowed"] = False
            security_result["issues"].extend(validation_result["issues"])
            security_result["security_level"] = validation_result["risk_level"]
        
        security_result["sanitized_data"] = validation_result.get("sanitized_data", request_data)
        
        # Rate limiting
        if source_ip or user_id:
            identifier = user_id or source_ip or "anonymous"
            
            # Check multiple rate limit rules
            for rule_name in ["api_requests", "evaluation_requests"]:
                rate_result = self.rate_limiter.check_rate_limit(rule_name, identifier, source_ip)
                if not rate_result["allowed"]:
                    security_result["allowed"] = False
                    security_result["issues"].append(f"Rate limit exceeded: {rule_name}")
                    security_result["rate_limit_info"] = rate_result
                    
                    self.audit_logger.log_event(
                        EventType.RATE_LIMIT,
                        SecurityLevel.MEDIUM,
                        f"Rate limit exceeded for {identifier} on rule {rule_name}",
                        source_ip=source_ip,
                        user_id=user_id,
                        metadata=rate_result
                    )
                    break
        
        # Suspicious IP check
        if source_ip and source_ip in self.audit_logger.suspicious_ips:
            security_result["allowed"] = False
            security_result["issues"].append("Request from suspicious IP address")
            security_result["security_level"] = SecurityLevel.HIGH
            
            self.audit_logger.log_event(
                EventType.SUSPICIOUS_ACTIVITY,
                SecurityLevel.HIGH,
                f"Request blocked from suspicious IP: {source_ip}",
                source_ip=source_ip,
                user_id=user_id
            )
        
        return security_result
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive security dashboard data."""
        return {
            "security_status": {
                "enabled": self.security_enabled,
                "strict_mode": self.strict_mode,
                "timestamp": datetime.utcnow().isoformat()
            },
            "rate_limiting": {
                "active_rules": len(self.rate_limiter.rules),
                "blocked_entities": sum(len(blocked) for blocked in self.rate_limiter.blocked_entities.values())
            },
            "security_events": self.audit_logger.get_security_summary(hours=24),
            "suspicious_activities": {
                "suspicious_ips": list(self.audit_logger.suspicious_ips),
                "recent_high_severity": len([
                    e for e in self.audit_logger.events
                    if e.level == SecurityLevel.HIGH
                    and (datetime.utcnow() - e.timestamp).total_seconds() < 3600
                ])
            }
        }


# Global security instance
_global_security = None

def get_security() -> ComprehensiveSecurity:
    """Get the global security instance."""
    global _global_security
    if _global_security is None:
        _global_security = ComprehensiveSecurity()
    return _global_security


def secure_endpoint(rate_limit_rule: Optional[str] = None):
    """Decorator to secure API endpoints."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            security = get_security()
            
            # Extract request info (this would be adapted based on your framework)
            request_data = kwargs.get('request_data', {})
            source_ip = kwargs.get('source_ip')
            user_id = kwargs.get('user_id')
            
            # Security check
            security_result = security.check_request_security(request_data, source_ip, user_id)
            
            if not security_result["allowed"]:
                raise SecurityError("Request blocked by security policy", security_result)
            
            # Update kwargs with sanitized data
            kwargs['request_data'] = security_result["sanitized_data"]
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            security = get_security()
            
            # Extract request info
            request_data = kwargs.get('request_data', {})
            source_ip = kwargs.get('source_ip')
            user_id = kwargs.get('user_id')
            
            # Security check
            security_result = security.check_request_security(request_data, source_ip, user_id)
            
            if not security_result["allowed"]:
                raise SecurityError("Request blocked by security policy", security_result)
            
            # Update kwargs with sanitized data
            kwargs['request_data'] = security_result["sanitized_data"]
            
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class SecurityError(Exception):
    """Security-related exception."""
    
    def __init__(self, message: str, security_result: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.security_result = security_result or {}