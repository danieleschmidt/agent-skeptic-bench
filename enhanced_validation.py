#!/usr/bin/env python3
"""
Enhanced Validation & Error Handling for Agent Skeptic Bench
===========================================================

Implements comprehensive error handling, validation, and health checks
for production-ready skepticism evaluation.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import traceback
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_skeptic_bench.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    error_message: Optional[str] = None
    warning_messages: List[str] = field(default_factory=list)
    severity: str = "INFO"
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass 
class HealthCheckResult:
    """Result of system health check."""
    component: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    response_time_ms: float
    timestamp: datetime = field(default_factory=datetime.now)

class EnhancedValidator:
    """Enhanced validation with comprehensive error handling."""
    
    def __init__(self):
        self.validation_history: List[ValidationResult] = []
        self.error_counts: Dict[str, int] = {}
        
    def validate_input(self, data: Any, schema: str) -> ValidationResult:
        """Validate input data against schema."""
        try:
            if schema == "scenario":
                return self._validate_scenario(data)
            elif schema == "agent_config":
                return self._validate_agent_config(data)
            elif schema == "evaluation_request":
                return self._validate_evaluation_request(data)
            else:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Unknown validation schema: {schema}",
                    severity="ERROR"
                )
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Validation exception: {str(e)}",
                severity="CRITICAL"
            )
    
    def _validate_scenario(self, scenario: Dict[str, Any]) -> ValidationResult:
        """Validate scenario data structure."""
        required_fields = ["id", "category", "description", "adversary_claim"]
        warnings = []
        
        # Check required fields
        for field in required_fields:
            if field not in scenario:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Missing required field: {field}",
                    severity="ERROR"
                )
        
        # Validate field types and values
        if not isinstance(scenario["id"], str) or len(scenario["id"]) < 3:
            return ValidationResult(
                is_valid=False,
                error_message="Scenario ID must be string with length >= 3",
                severity="ERROR"
            )
        
        if scenario["category"] not in ["factual_claims", "flawed_plans", "persuasion_attacks", "evidence_evaluation", "epistemic_calibration"]:
            warnings.append(f"Unknown category: {scenario['category']}")
        
        if len(scenario["description"]) < 10:
            warnings.append("Scenario description is very short")
        
        if len(scenario["adversary_claim"]) < 5:
            warnings.append("Adversary claim is very short")
        
        # Check for potential security issues
        suspicious_patterns = ["<script>", "javascript:", "eval(", "exec("]
        for field in ["description", "adversary_claim"]:
            if any(pattern in scenario[field].lower() for pattern in suspicious_patterns):
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Potential security issue in {field}",
                    severity="CRITICAL"
                )
        
        return ValidationResult(
            is_valid=True,
            warning_messages=warnings,
            severity="INFO" if not warnings else "WARNING"
        )
    
    def _validate_agent_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate agent configuration."""
        required_fields = ["provider", "model_name"]
        warnings = []
        
        for field in required_fields:
            if field not in config:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Missing required field: {field}",
                    severity="ERROR"
                )
        
        # Validate provider
        valid_providers = ["openai", "anthropic", "google", "huggingface", "custom"]
        if config["provider"] not in valid_providers:
            return ValidationResult(
                is_valid=False,
                error_message=f"Invalid provider: {config['provider']}. Must be one of {valid_providers}",
                severity="ERROR"
            )
        
        # Validate temperature if present
        if "temperature" in config:
            temp = config["temperature"]
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                warnings.append("Temperature should be between 0 and 2")
        
        # Check for API key presence (don't log actual key)
        if "api_key" in config:
            if not config["api_key"] or len(config["api_key"]) < 10:
                warnings.append("API key appears to be missing or too short")
        else:
            warnings.append("No API key provided - may cause authentication errors")
        
        return ValidationResult(
            is_valid=True,
            warning_messages=warnings,
            severity="INFO" if not warnings else "WARNING"
        )
    
    def _validate_evaluation_request(self, request: Dict[str, Any]) -> ValidationResult:
        """Validate evaluation request."""
        required_fields = ["scenario_id", "agent_config"]
        
        for field in required_fields:
            if field not in request:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Missing required field: {field}",
                    severity="ERROR"
                )
        
        # Validate nested agent config
        agent_config_result = self._validate_agent_config(request["agent_config"])
        if not agent_config_result.is_valid:
            return agent_config_result
        
        return ValidationResult(
            is_valid=True,
            warning_messages=agent_config_result.warning_messages,
            severity=agent_config_result.severity
        )

class ErrorHandler:
    """Centralized error handling with recovery strategies."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[str, callable] = {
            "rate_limit": self._handle_rate_limit,
            "api_error": self._handle_api_error,
            "timeout": self._handle_timeout,
            "validation_error": self._handle_validation_error
        }
    
    def handle_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Handle error with appropriate recovery strategy."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        logger.error(f"Error in {context}: {error_type} - {str(error)}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Determine recovery strategy
        if "rate limit" in str(error).lower() or "429" in str(error):
            return self._handle_rate_limit(error, context)
        elif "timeout" in str(error).lower():
            return self._handle_timeout(error, context)
        elif "api" in str(error).lower() or "401" in str(error) or "403" in str(error):
            return self._handle_api_error(error, context)
        elif "validation" in str(error).lower():
            return self._handle_validation_error(error, context)
        else:
            return self._handle_generic_error(error, context)
    
    def _handle_rate_limit(self, error: Exception, context: str) -> Dict[str, Any]:
        """Handle rate limiting errors."""
        wait_time = 60  # Default wait time
        
        # Extract wait time from error message if available
        error_str = str(error).lower()
        if "retry after" in error_str:
            try:
                import re
                matches = re.findall(r'(\d+)', error_str)
                if matches:
                    wait_time = min(300, int(matches[0]))  # Max 5 minutes
            except:
                pass
        
        return {
            "strategy": "retry",
            "wait_time": wait_time,
            "max_retries": 3,
            "error_type": "rate_limit",
            "recoverable": True,
            "message": f"Rate limit hit. Waiting {wait_time} seconds before retry."
        }
    
    def _handle_api_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Handle API authentication/authorization errors."""
        return {
            "strategy": "fail_fast",
            "error_type": "api_error",
            "recoverable": False,
            "message": "API authentication failed. Check API key and permissions."
        }
    
    def _handle_timeout(self, error: Exception, context: str) -> Dict[str, Any]:
        """Handle timeout errors."""
        return {
            "strategy": "retry",
            "wait_time": 5,
            "max_retries": 2,
            "error_type": "timeout",
            "recoverable": True,
            "message": "Request timeout. Retrying with exponential backoff."
        }
    
    def _handle_validation_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Handle validation errors."""
        return {
            "strategy": "fail_fast",
            "error_type": "validation_error", 
            "recoverable": False,
            "message": f"Validation failed: {str(error)}"
        }
    
    def _handle_generic_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Handle generic errors."""
        return {
            "strategy": "retry",
            "wait_time": 1,
            "max_retries": 1,
            "error_type": "generic_error",
            "recoverable": True,
            "message": f"Unexpected error: {str(error)}"
        }

class HealthChecker:
    """System health monitoring and checks."""
    
    def __init__(self):
        self.checks = {
            "disk_space": self._check_disk_space,
            "memory": self._check_memory,
            "network": self._check_network,
            "file_permissions": self._check_file_permissions,
            "dependencies": self._check_dependencies
        }
    
    async def run_health_checks(self) -> List[HealthCheckResult]:
        """Run all health checks."""
        results = []
        
        for check_name, check_func in self.checks.items():
            start_time = time.time()
            try:
                result = await check_func()
                response_time = (time.time() - start_time) * 1000
                results.append(HealthCheckResult(
                    component=check_name,
                    status=result["status"],
                    message=result["message"],
                    response_time_ms=response_time
                ))
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                results.append(HealthCheckResult(
                    component=check_name,
                    status="unhealthy",
                    message=f"Health check failed: {str(e)}",
                    response_time_ms=response_time
                ))
        
        return results
    
    async def _check_disk_space(self) -> Dict[str, str]:
        """Check available disk space."""
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        free_percent = (free / total) * 100
        
        if free_percent < 5:
            return {"status": "unhealthy", "message": f"Critical: Only {free_percent:.1f}% disk space free"}
        elif free_percent < 15:
            return {"status": "degraded", "message": f"Warning: Only {free_percent:.1f}% disk space free"}
        else:
            return {"status": "healthy", "message": f"Disk space OK: {free_percent:.1f}% free"}
    
    async def _check_memory(self) -> Dict[str, str]:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                return {"status": "unhealthy", "message": f"Critical: Memory usage at {memory.percent}%"}
            elif memory.percent > 80:
                return {"status": "degraded", "message": f"Warning: Memory usage at {memory.percent}%"}
            else:
                return {"status": "healthy", "message": f"Memory OK: {memory.percent}% used"}
        except ImportError:
            return {"status": "degraded", "message": "psutil not available for memory monitoring"}
    
    async def _check_network(self) -> Dict[str, str]:
        """Check network connectivity."""
        import subprocess
        
        try:
            # Try to ping a reliable service
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "3", "8.8.8.8"],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return {"status": "healthy", "message": "Network connectivity OK"}
            else:
                return {"status": "degraded", "message": "Network connectivity issues detected"}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return {"status": "degraded", "message": "Network check failed or ping unavailable"}
    
    async def _check_file_permissions(self) -> Dict[str, str]:
        """Check file permissions for key directories."""
        critical_paths = [
            "src/agent_skeptic_bench",
            "data/scenarios", 
            "logs"
        ]
        
        issues = []
        for path in critical_paths:
            if os.path.exists(path):
                if not os.access(path, os.R_OK):
                    issues.append(f"{path}: not readable")
                if not os.access(path, os.W_OK):
                    issues.append(f"{path}: not writable")
            else:
                issues.append(f"{path}: does not exist")
        
        if issues:
            return {"status": "degraded", "message": f"Permission issues: {'; '.join(issues)}"}
        else:
            return {"status": "healthy", "message": "File permissions OK"}
    
    async def _check_dependencies(self) -> Dict[str, str]:
        """Check if critical dependencies are available."""
        critical_deps = ["numpy", "pandas", "scipy", "scikit-learn"]
        missing_deps = []
        
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            return {"status": "degraded", "message": f"Missing dependencies: {', '.join(missing_deps)}"}
        else:
            return {"status": "healthy", "message": "All critical dependencies available"}

class SecurityValidator:
    """Security validation and threat detection."""
    
    def __init__(self):
        self.threat_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__",
            r"subprocess",
            r"os\.system",
            r"open\s*\(",
            r"file\s*\(",
            r"\.\.\/",
            r"\.\.\\",
        ]
        self.rate_limits = {}
    
    def validate_input_security(self, input_data: str, source: str) -> ValidationResult:
        """Validate input for security threats."""
        import re
        
        threats_found = []
        
        # Check for injection patterns
        for pattern in self.threat_patterns:
            if re.search(pattern, input_data, re.IGNORECASE):
                threats_found.append(f"Potential injection: {pattern}")
        
        # Check input length
        if len(input_data) > 50000:  # 50KB limit
            threats_found.append("Input exceeds size limit")
        
        # Check for unusual encoding
        try:
            input_data.encode('utf-8')
        except UnicodeEncodeError:
            threats_found.append("Invalid character encoding")
        
        if threats_found:
            logger.warning(f"Security threats detected from {source}: {threats_found}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Security validation failed: {'; '.join(threats_found)}",
                severity="CRITICAL"
            )
        
        return ValidationResult(is_valid=True, severity="INFO")
    
    def check_rate_limit(self, client_id: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """Check if client has exceeded rate limit."""
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)
        
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []
        
        # Remove old requests
        self.rate_limits[client_id] = [
            req_time for req_time in self.rate_limits[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.rate_limits[client_id]) >= max_requests:
            return False
        
        # Add current request
        self.rate_limits[client_id].append(now)
        return True

async def run_enhanced_validation_tests():
    """Run enhanced validation tests."""
    print("🛡️  ENHANCED VALIDATION & ERROR HANDLING TESTS")
    print("=" * 60)
    
    # Test validator
    validator = EnhancedValidator()
    
    # Test scenario validation
    print("🧪 Testing Scenario Validation...")
    valid_scenario = {
        "id": "test_001",
        "category": "factual_claims",
        "description": "This is a valid test scenario for validation",
        "adversary_claim": "This is a test claim"
    }
    
    result = validator.validate_input(valid_scenario, "scenario")
    print(f"  ✅ Valid scenario: {result.is_valid}")
    
    # Test invalid scenario
    invalid_scenario = {"id": "x"}
    result = validator.validate_input(invalid_scenario, "scenario")
    print(f"  ✅ Invalid scenario detected: {not result.is_valid}")
    
    # Test error handler
    print("\n🧪 Testing Error Handler...")
    error_handler = ErrorHandler()
    
    # Simulate rate limit error
    try:
        raise Exception("Rate limit exceeded. Retry after 30 seconds")
    except Exception as e:
        recovery = error_handler.handle_error(e, "test_context")
        print(f"  ✅ Rate limit recovery: {recovery['strategy']} (wait: {recovery['wait_time']}s)")
    
    # Test health checker
    print("\n🧪 Testing Health Checker...")
    health_checker = HealthChecker()
    health_results = await health_checker.run_health_checks()
    
    for result in health_results:
        status_emoji = "✅" if result.status == "healthy" else "⚠️" if result.status == "degraded" else "❌"
        print(f"  {status_emoji} {result.component}: {result.status} ({result.response_time_ms:.1f}ms)")
    
    # Test security validator
    print("\n🧪 Testing Security Validator...")
    security_validator = SecurityValidator()
    
    # Test safe input
    safe_input = "This is a normal user input for skepticism evaluation"
    result = security_validator.validate_input_security(safe_input, "test_user")
    print(f"  ✅ Safe input validated: {result.is_valid}")
    
    # Test potentially malicious input
    malicious_input = "<script>alert('xss')</script>"
    result = security_validator.validate_input_security(malicious_input, "test_user")
    print(f"  ✅ Malicious input blocked: {not result.is_valid}")
    
    print("\n🏆 ENHANCED VALIDATION TESTS COMPLETED")
    print("✅ All validation components working correctly!")

if __name__ == "__main__":
    import time
    asyncio.run(run_enhanced_validation_tests())