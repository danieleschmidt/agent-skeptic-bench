"""Robustness Framework for Agent Skeptic Bench.

Comprehensive error handling, validation, security, and reliability features
for production-ready skepticism evaluation systems.
"""

import asyncio
import functools
import hashlib
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Generic
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class SecurityAuditEntry:
    """Security audit log entry."""
    operation: str
    user_id: Optional[str]
    success: bool
    security_level: SecurityLevel
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    ip_address: Optional[str] = None


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    parameters: Dict[str, Any]
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0


class RobustnessError(Exception):
    """Base exception for robustness framework."""
    
    def __init__(self, message: str, error_code: str, context: Optional[ErrorContext] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or ErrorContext("unknown", {}, None, None)


class ValidationError(RobustnessError):
    """Exception for validation failures."""
    pass


class SecurityError(RobustnessError):
    """Exception for security violations."""
    pass


class RateLimitError(RobustnessError):
    """Exception for rate limiting violations."""
    pass


class RetryableError(RobustnessError):
    """Exception for operations that can be retried."""
    pass


class InputValidator:
    """Comprehensive input validation framework."""
    
    def __init__(self):
        """Initialize input validator."""
        self.validation_rules: Dict[str, List[Callable]] = {}
        self.sanitization_rules: Dict[str, List[Callable]] = {}
        
    def add_validation_rule(self, field_name: str, rule: Callable[[Any], ValidationResult]):
        """Add a validation rule for a specific field."""
        if field_name not in self.validation_rules:
            self.validation_rules[field_name] = []
        self.validation_rules[field_name].append(rule)
    
    def add_sanitization_rule(self, field_name: str, rule: Callable[[Any], Any]):
        """Add a sanitization rule for a specific field."""
        if field_name not in self.sanitization_rules:
            self.sanitization_rules[field_name] = []
        self.sanitization_rules[field_name].append(rule)
    
    def validate(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate input data against all rules."""
        results = []
        
        for field_name, value in data.items():
            if field_name in self.validation_rules:
                for rule in self.validation_rules[field_name]:
                    try:
                        result = rule(value)
                        results.append(result)
                        
                        if not result.is_valid and result.severity == ValidationSeverity.CRITICAL:
                            # Stop validation on critical errors
                            return results
                            
                    except Exception as e:
                        logger.error(f"Validation rule failed for {field_name}: {e}")
                        results.append(ValidationResult(
                            is_valid=False,
                            severity=ValidationSeverity.ERROR,
                            message=f"Validation rule execution failed: {str(e)}",
                            details={"field": field_name, "error": str(e)}
                        ))
        
        return results
    
    def sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data using sanitization rules."""
        sanitized = data.copy()
        
        for field_name, value in data.items():
            if field_name in self.sanitization_rules:
                for rule in self.sanitization_rules[field_name]:
                    try:
                        sanitized[field_name] = rule(sanitized[field_name])
                    except Exception as e:
                        logger.warning(f"Sanitization rule failed for {field_name}: {e}")
                        # Continue with original value if sanitization fails
        
        return sanitized
    
    def validate_and_sanitize(self, data: Dict[str, Any]) -> tuple[Dict[str, Any], List[ValidationResult]]:
        """Validate and sanitize input data."""
        # First sanitize
        sanitized_data = self.sanitize(data)
        
        # Then validate sanitized data
        validation_results = self.validate(sanitized_data)
        
        return sanitized_data, validation_results


class SecurityManager:
    """Comprehensive security management system."""
    
    def __init__(self):
        """Initialize security manager."""
        self.audit_log: List[SecurityAuditEntry] = []
        self.blocked_ips: set = set()
        self.suspicious_patterns: List[str] = [
            r'<script[^>]*>.*?</script>',  # XSS attempts
            r'union\s+select',  # SQL injection
            r'../../../',  # Path traversal
            r'eval\s*\(',  # Code injection
        ]
        
    def validate_security_level(self, required_level: SecurityLevel, current_user: Optional[str] = None) -> bool:
        """Validate if operation meets required security level."""
        # Simplified security check
        if required_level == SecurityLevel.CRITICAL and not current_user:
            return False
        
        # Check if user is in blocked list (simplified)
        if current_user and current_user.startswith('blocked_'):
            return False
        
        return True
    
    def scan_for_threats(self, text: str) -> List[str]:
        """Scan text for security threats."""
        threats = []
        
        import re
        for pattern in self.suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                threats.append(f"Suspicious pattern detected: {pattern}")
        
        # Check for excessive length (potential DoS)
        if len(text) > 100000:
            threats.append("Input exceeds maximum length limit")
        
        # Check for binary data
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            threats.append("Invalid character encoding detected")
        
        return threats
    
    def audit_operation(self, entry: SecurityAuditEntry):
        """Add entry to security audit log."""
        self.audit_log.append(entry)
        
        # Log security violations
        if not entry.success:
            logger.warning(f"Security audit: {entry.operation} failed for user {entry.user_id}")
    
    def check_rate_limit(self, user_id: str, operation: str, window_minutes: int = 60, max_requests: int = 100) -> bool:
        """Check if user has exceeded rate limits."""
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        # Count recent requests from this user for this operation
        recent_requests = [
            entry for entry in self.audit_log
            if (entry.user_id == user_id and 
                entry.operation == operation and
                entry.timestamp >= window_start)
        ]
        
        return len(recent_requests) < max_requests


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self):
        """Initialize error handler."""
        self.error_log: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[str, Callable] = {}
        
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy
    
    async def handle_error(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Handle an error with appropriate recovery strategy."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'timestamp': time.time(),
            'traceback': traceback.format_exc()
        }
        
        self.error_log.append(error_info)
        
        # Log the error
        logger.error(f"Error in {context.operation}: {error_info['error_message']}")
        
        # Try to recover
        error_type = type(error).__name__
        if error_type in self.recovery_strategies:
            try:
                recovery_result = await self.recovery_strategies[error_type](error, context)
                logger.info(f"Recovery successful for {error_type}")
                return recovery_result
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {error_type}: {recovery_error}")
        
        # Re-raise if no recovery possible
        raise error
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        if not self.error_log:
            return {"total_errors": 0}
        
        error_types = {}
        for error in self.error_log:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        recent_errors = [
            error for error in self.error_log
            if time.time() - error['timestamp'] < 3600  # Last hour
        ]
        
        return {
            "total_errors": len(self.error_log),
            "error_types": error_types,
            "recent_errors": len(recent_errors),
            "most_common_error": max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }


class CircuitBreaker:
    """Circuit breaker pattern for resilient operations."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if self.state == "open":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "half-open"
                    logger.info("Circuit breaker entering half-open state")
                else:
                    raise RobustnessError(
                        "Circuit breaker is open",
                        "CIRCUIT_BREAKER_OPEN",
                        ErrorContext("circuit_breaker", {}, None, None)
                    )
            
            try:
                result = await func(*args, **kwargs)
                
                if self.state == "half-open":
                    self.state = "closed"
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to closed state")
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
                
                raise e
        
        return wrapper


class RetryManager:
    """Intelligent retry mechanism with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        """Initialize retry manager."""
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to add retry logic to functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except RetryableError as e:
                    last_exception = e
                    
                    if attempt == self.max_retries:
                        logger.error(f"Max retries ({self.max_retries}) exceeded for {func.__name__}")
                        break
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    jitter = delay * 0.1 * (2 * hash(str(args)) % 100 / 100 - 1)  # ±10% jitter
                    actual_delay = delay + jitter
                    
                    logger.warning(f"Retrying {func.__name__} in {actual_delay:.2f}s (attempt {attempt + 1})")
                    await asyncio.sleep(actual_delay)
                    
                except Exception as e:
                    # Non-retryable error
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise e
            
            # If we get here, all retries failed
            raise last_exception
        
        return wrapper


class HealthChecker:
    """System health monitoring and checking."""
    
    def __init__(self):
        """Initialize health checker."""
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        
    def register_health_check(self, name: str, check_func: Callable):
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        overall_healthy = True
        results = {}
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = await check_func()
                check_time = time.time() - start_time
                
                check_result = {
                    "healthy": result.get("healthy", True),
                    "message": result.get("message", "OK"),
                    "details": result.get("details", {}),
                    "check_time": check_time,
                    "timestamp": time.time()
                }
                
                if not check_result["healthy"]:
                    overall_healthy = False
                
                results[name] = check_result
                
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = {
                    "healthy": False,
                    "message": f"Health check failed: {str(e)}",
                    "details": {"error": str(e)},
                    "check_time": 0,
                    "timestamp": time.time()
                }
                overall_healthy = False
        
        self.health_status = results
        
        return {
            "overall_healthy": overall_healthy,
            "checks": results,
            "timestamp": time.time()
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of system health."""
        if not self.health_status:
            return {"status": "unknown", "message": "No health checks run yet"}
        
        healthy_checks = sum(1 for check in self.health_status.values() if check["healthy"])
        total_checks = len(self.health_status)
        
        if healthy_checks == total_checks:
            status = "healthy"
        elif healthy_checks > 0:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "healthy_checks": healthy_checks,
            "total_checks": total_checks,
            "last_check": max(check["timestamp"] for check in self.health_status.values()),
            "failing_checks": [
                name for name, check in self.health_status.items()
                if not check["healthy"]
            ]
        }


class RobustnessFramework:
    """Main robustness framework coordinating all components."""
    
    def __init__(self):
        """Initialize robustness framework."""
        self.input_validator = InputValidator()
        self.security_manager = SecurityManager()
        self.error_handler = ErrorHandler()
        self.health_checker = HealthChecker()
        
        self._setup_default_rules()
        self._setup_default_health_checks()
        self._setup_default_recovery_strategies()
    
    def _setup_default_rules(self):
        """Setup default validation and sanitization rules."""
        # Text input validation
        def validate_text_length(text: str) -> ValidationResult:
            if len(text) > 50000:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Text exceeds maximum length (50,000 characters)"
                )
            return ValidationResult(is_valid=True, severity=ValidationSeverity.INFO, message="OK")
        
        def validate_text_encoding(text: str) -> ValidationResult:
            try:
                text.encode('utf-8')
                return ValidationResult(is_valid=True, severity=ValidationSeverity.INFO, message="OK")
            except UnicodeEncodeError:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Invalid character encoding"
                )
        
        # Numeric validation
        def validate_skepticism_range(value: float) -> ValidationResult:
            if not 0.0 <= value <= 1.0:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Skepticism value must be between 0.0 and 1.0"
                )
            return ValidationResult(is_valid=True, severity=ValidationSeverity.INFO, message="OK")
        
        # Sanitization rules
        def sanitize_text(text: str) -> str:
            # Remove potential XSS patterns
            import re
            sanitized = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
            sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
            return sanitized.strip()
        
        def sanitize_numeric(value: Any) -> float:
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0
        
        # Register rules
        self.input_validator.add_validation_rule("text", validate_text_length)
        self.input_validator.add_validation_rule("text", validate_text_encoding)
        self.input_validator.add_validation_rule("description", validate_text_length)
        self.input_validator.add_validation_rule("description", validate_text_encoding)
        self.input_validator.add_validation_rule("skepticism_level", validate_skepticism_range)
        
        self.input_validator.add_sanitization_rule("text", sanitize_text)
        self.input_validator.add_sanitization_rule("description", sanitize_text)
        self.input_validator.add_sanitization_rule("skepticism_level", sanitize_numeric)
    
    def _setup_default_health_checks(self):
        """Setup default system health checks."""
        async def check_memory_usage():
            """Check system memory usage."""
            import psutil
            try:
                memory = psutil.virtual_memory()
                healthy = memory.percent < 90
                return {
                    "healthy": healthy,
                    "message": f"Memory usage: {memory.percent:.1f}%",
                    "details": {
                        "percent": memory.percent,
                        "available": memory.available,
                        "total": memory.total
                    }
                }
            except ImportError:
                # psutil not available, simulate check
                return {
                    "healthy": True,
                    "message": "Memory check not available",
                    "details": {}
                }
        
        async def check_error_rate():
            """Check system error rate."""
            stats = self.error_handler.get_error_statistics()
            recent_errors = stats.get("recent_errors", 0)
            healthy = recent_errors < 10  # Less than 10 errors in last hour
            
            return {
                "healthy": healthy,
                "message": f"Recent errors: {recent_errors}",
                "details": stats
            }
        
        async def check_security_violations():
            """Check for recent security violations."""
            recent_violations = len([
                entry for entry in self.security_manager.audit_log
                if not entry.success and time.time() - entry.timestamp < 3600
            ])
            
            healthy = recent_violations < 5
            
            return {
                "healthy": healthy,
                "message": f"Security violations in last hour: {recent_violations}",
                "details": {"violations": recent_violations}
            }
        
        self.health_checker.register_health_check("memory", check_memory_usage)
        self.health_checker.register_health_check("errors", check_error_rate)
        self.health_checker.register_health_check("security", check_security_violations)
    
    def _setup_default_recovery_strategies(self):
        """Setup default error recovery strategies."""
        async def retry_on_timeout(error: Exception, context: ErrorContext):
            """Recovery strategy for timeout errors."""
            if context.retry_count < 3:
                context.retry_count += 1
                logger.info(f"Retrying operation {context.operation} (attempt {context.retry_count})")
                await asyncio.sleep(min(2 ** context.retry_count, 10))  # Exponential backoff
                return "RETRY"
            else:
                logger.error(f"Max retries exceeded for {context.operation}")
                return None
        
        async def fallback_on_validation_error(error: Exception, context: ErrorContext):
            """Recovery strategy for validation errors."""
            logger.warning(f"Using fallback for validation error in {context.operation}")
            # Return safe default values
            return {
                "skepticism_level": 0.5,
                "confidence": 0.1,
                "message": "Fallback response due to validation error"
            }
        
        self.error_handler.register_recovery_strategy("TimeoutError", retry_on_timeout)
        self.error_handler.register_recovery_strategy("ValidationError", fallback_on_validation_error)
    
    @asynccontextmanager
    async def robust_operation(self, 
                             operation_name: str,
                             user_id: Optional[str] = None,
                             security_level: SecurityLevel = SecurityLevel.MEDIUM,
                             input_data: Optional[Dict[str, Any]] = None):
        """Context manager for robust operation execution."""
        context = ErrorContext(operation_name, input_data or {}, user_id, None)
        
        # Security check
        if not self.security_manager.validate_security_level(security_level, user_id):
            raise SecurityError(
                f"Insufficient security level for operation {operation_name}",
                "INSUFFICIENT_SECURITY",
                context
            )
        
        # Rate limiting check
        if user_id and not self.security_manager.check_rate_limit(user_id, operation_name):
            raise RateLimitError(
                f"Rate limit exceeded for user {user_id}",
                "RATE_LIMIT_EXCEEDED",
                context
            )
        
        # Input validation
        if input_data:
            threats = self.security_manager.scan_for_threats(str(input_data))
            if threats:
                raise SecurityError(
                    f"Security threats detected: {threats}",
                    "SECURITY_THREAT",
                    context
                )
            
            sanitized_data, validation_results = self.input_validator.validate_and_sanitize(input_data)
            
            critical_errors = [r for r in validation_results if r.severity == ValidationSeverity.CRITICAL]
            if critical_errors:
                raise ValidationError(
                    f"Critical validation errors: {[e.message for e in critical_errors]}",
                    "VALIDATION_CRITICAL",
                    context
                )
        
        # Audit log entry
        audit_entry = SecurityAuditEntry(
            operation=operation_name,
            user_id=user_id,
            success=True,  # Will be updated on failure
            security_level=security_level,
            details={"input_data_size": len(str(input_data)) if input_data else 0}
        )
        
        try:
            yield context
            
        except Exception as e:
            audit_entry.success = False
            self.security_manager.audit_operation(audit_entry)
            
            # Try error recovery
            await self.error_handler.handle_error(e, context)
            
        else:
            self.security_manager.audit_operation(audit_entry)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_status = await self.health_checker.run_health_checks()
        error_stats = self.error_handler.get_error_statistics()
        
        return {
            "health": health_status,
            "errors": error_stats,
            "security": {
                "total_audits": len(self.security_manager.audit_log),
                "blocked_ips": len(self.security_manager.blocked_ips)
            },
            "timestamp": time.time()
        }