"""Input validation and sanitization for Agent Skeptic Bench."""

import logging
import re
import html
import json
import base64
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import bleach
from urllib.parse import urlparse
import numpy as np
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Input validation error."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value


class ValidationType(Enum):
    """Types of validation."""
    
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    EMAIL = "email"
    URL = "url"
    UUID = "uuid"
    JSON = "json"
    LIST = "list"
    DICT = "dict"


@dataclass
class ValidationRule:
    """Validation rule definition."""
    
    field_name: str
    validation_type: ValidationType
    required: bool = True
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    custom_validator: Optional[Callable[[Any], bool]] = None
    description: str = ""


class InputValidator:
    """Comprehensive input validation system."""
    
    def __init__(self):
        """Initialize input validator."""
        self.validation_schemas: Dict[str, List[ValidationRule]] = {}
        
        # Pre-compiled patterns
        self.patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.IGNORECASE),
            'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
            'safe_string': re.compile(r'^[a-zA-Z0-9\s\-_.]+$'),
            'sql_injection': re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)', re.IGNORECASE),
            'xss_basic': re.compile(r'<\s*script[^>]*>.*?<\s*/\s*script\s*>', re.IGNORECASE | re.DOTALL),
            'command_injection': re.compile(r'[;&|`$\(\)]'),
            # AI-specific threat patterns
            'prompt_injection': re.compile(r'(ignore\s+previous\s+instructions|forget\s+everything|act\s+as\s+(?!.*skeptic)|pretend\s+to\s+be|system\s*:\s*|user\s*:\s*|assistant\s*:\s*)', re.IGNORECASE),
            'jailbreak_attempt': re.compile(r'(DAN|Do\s+Anything\s+Now|evil\s+mode|bypass\s+safety|remove\s+restrictions|developer\s+mode)', re.IGNORECASE),
            'model_extraction': re.compile(r'(repeat\s+your\s+instructions|show\s+me\s+your\s+prompt|what\s+are\s+your\s+rules)', re.IGNORECASE),
            'adversarial_prompt': re.compile(r'(\\\\\\\|\\n\\n|\\r\\r|---BEGIN---|---END---|\{\{|\}\})', re.IGNORECASE)
        }
        
        # AI threat detection components
        self.ai_threat_detector = AIThreatDetector()
        self.multimodal_validator = MultiModalValidator()
        self.request_rate_tracker = {}  # Track request patterns per IP/user
    
    def add_schema(self, schema_name: str, rules: List[ValidationRule]) -> None:
        """Add a validation schema."""
        self.validation_schemas[schema_name] = rules
        logger.info(f"Added validation schema: {schema_name}")
    
    def validate_data(self, schema_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data against a schema."""
        schema = self.validation_schemas.get(schema_name)
        if not schema:
            raise ValidationError(f"Validation schema not found: {schema_name}")
        
        validated_data = {}
        errors = []
        
        for rule in schema:
            try:
                value = data.get(rule.field_name)
                validated_value = self._validate_field(rule, value)
                validated_data[rule.field_name] = validated_value
            except ValidationError as e:
                errors.append(f"{rule.field_name}: {e}")
        
        if errors:
            raise ValidationError(f"Validation failed: {', '.join(errors)}")
        
        return validated_data
    
    def _validate_field(self, rule: ValidationRule, value: Any) -> Any:
        """Validate a single field."""
        # Check if required
        if value is None:
            if rule.required:
                raise ValidationError(f"Field '{rule.field_name}' is required")
            return None
        
        # Type-specific validation
        if rule.validation_type == ValidationType.STRING:
            return self._validate_string(rule, value)
        elif rule.validation_type == ValidationType.INTEGER:
            return self._validate_integer(rule, value)
        elif rule.validation_type == ValidationType.FLOAT:
            return self._validate_float(rule, value)
        elif rule.validation_type == ValidationType.BOOLEAN:
            return self._validate_boolean(rule, value)
        elif rule.validation_type == ValidationType.EMAIL:
            return self._validate_email(rule, value)
        elif rule.validation_type == ValidationType.URL:
            return self._validate_url(rule, value)
        elif rule.validation_type == ValidationType.UUID:
            return self._validate_uuid(rule, value)
        elif rule.validation_type == ValidationType.JSON:
            return self._validate_json(rule, value)
        elif rule.validation_type == ValidationType.LIST:
            return self._validate_list(rule, value)
        elif rule.validation_type == ValidationType.DICT:
            return self._validate_dict(rule, value)
        else:
            raise ValidationError(f"Unknown validation type: {rule.validation_type}")
    
    def _validate_string(self, rule: ValidationRule, value: Any) -> str:
        """Validate string value."""
        if not isinstance(value, str):
            try:
                value = str(value)
            except:
                raise ValidationError("Value must be a string")
        
        # Length validation
        if rule.min_length is not None and len(value) < rule.min_length:
            raise ValidationError(f"String must be at least {rule.min_length} characters")
        
        if rule.max_length is not None and len(value) > rule.max_length:
            raise ValidationError(f"String must be at most {rule.max_length} characters")
        
        # Pattern validation
        if rule.pattern:
            if not re.match(rule.pattern, value):
                raise ValidationError(f"String does not match required pattern")
        
        # Allowed values
        if rule.allowed_values and value not in rule.allowed_values:
            raise ValidationError(f"Value must be one of: {rule.allowed_values}")
        
        # Security checks
        if self._contains_malicious_content(value):
            raise ValidationError("String contains potentially malicious content")
        
        # Custom validator
        if rule.custom_validator and not rule.custom_validator(value):
            raise ValidationError("Custom validation failed")
        
        return value
    
    def _validate_integer(self, rule: ValidationRule, value: Any) -> int:
        """Validate integer value."""
        try:
            if isinstance(value, str):
                value = int(value)
            elif not isinstance(value, int):
                raise ValueError()
        except (ValueError, TypeError):
            raise ValidationError("Value must be an integer")
        
        # Range validation
        if rule.min_value is not None and value < rule.min_value:
            raise ValidationError(f"Value must be at least {rule.min_value}")
        
        if rule.max_value is not None and value > rule.max_value:
            raise ValidationError(f"Value must be at most {rule.max_value}")
        
        # Allowed values
        if rule.allowed_values and value not in rule.allowed_values:
            raise ValidationError(f"Value must be one of: {rule.allowed_values}")
        
        return value
    
    def _validate_float(self, rule: ValidationRule, value: Any) -> float:
        """Validate float value."""
        try:
            if isinstance(value, str):
                value = float(value)
            elif not isinstance(value, (int, float)):
                raise ValueError()
        except (ValueError, TypeError):
            raise ValidationError("Value must be a number")
        
        # Range validation
        if rule.min_value is not None and value < rule.min_value:
            raise ValidationError(f"Value must be at least {rule.min_value}")
        
        if rule.max_value is not None and value > rule.max_value:
            raise ValidationError(f"Value must be at most {rule.max_value}")
        
        return float(value)
    
    def _validate_boolean(self, rule: ValidationRule, value: Any) -> bool:
        """Validate boolean value."""
        if isinstance(value, bool):
            return value
        elif isinstance(value, str):
            if value.lower() in ('true', '1', 'yes', 'on'):
                return True
            elif value.lower() in ('false', '0', 'no', 'off'):
                return False
            else:
                raise ValidationError("Boolean value must be true/false")
        elif isinstance(value, (int, float)):
            return bool(value)
        else:
            raise ValidationError("Value must be a boolean")
    
    def _validate_email(self, rule: ValidationRule, value: Any) -> str:
        """Validate email address."""
        if not isinstance(value, str):
            raise ValidationError("Email must be a string")
        
        if not self.patterns['email'].match(value):
            raise ValidationError("Invalid email format")
        
        # Additional security checks
        if len(value) > 254:  # RFC 5321 limit
            raise ValidationError("Email address too long")
        
        return value.lower()
    
    def _validate_url(self, rule: ValidationRule, value: Any) -> str:
        """Validate URL."""
        if not isinstance(value, str):
            raise ValidationError("URL must be a string")
        
        try:
            parsed = urlparse(value)
            if not parsed.scheme or not parsed.netloc:
                raise ValidationError("Invalid URL format")
            
            # Security check - only allow certain schemes
            allowed_schemes = ['http', 'https', 'ftp', 'ftps']
            if parsed.scheme.lower() not in allowed_schemes:
                raise ValidationError(f"URL scheme must be one of: {allowed_schemes}")
            
        except Exception:
            raise ValidationError("Invalid URL format")
        
        return value
    
    def _validate_uuid(self, rule: ValidationRule, value: Any) -> str:
        """Validate UUID."""
        if not isinstance(value, str):
            raise ValidationError("UUID must be a string")
        
        if not self.patterns['uuid'].match(value):
            raise ValidationError("Invalid UUID format")
        
        return value.lower()
    
    def _validate_json(self, rule: ValidationRule, value: Any) -> Any:
        """Validate JSON."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                raise ValidationError("Invalid JSON format")
        elif isinstance(value, (dict, list)):
            return value
        else:
            raise ValidationError("Value must be valid JSON")
    
    def _validate_list(self, rule: ValidationRule, value: Any) -> List[Any]:
        """Validate list."""
        if not isinstance(value, list):
            raise ValidationError("Value must be a list")
        
        # Length validation
        if rule.min_length is not None and len(value) < rule.min_length:
            raise ValidationError(f"List must have at least {rule.min_length} items")
        
        if rule.max_length is not None and len(value) > rule.max_length:
            raise ValidationError(f"List must have at most {rule.max_length} items")
        
        return value
    
    def _validate_dict(self, rule: ValidationRule, value: Any) -> Dict[str, Any]:
        """Validate dictionary."""
        if not isinstance(value, dict):
            raise ValidationError("Value must be a dictionary")
        
        # Length validation (number of keys)
        if rule.min_length is not None and len(value) < rule.min_length:
            raise ValidationError(f"Dictionary must have at least {rule.min_length} keys")
        
        if rule.max_length is not None and len(value) > rule.max_length:
            raise ValidationError(f"Dictionary must have at most {rule.max_length} keys")
        
        return value
    
    def _contains_malicious_content(self, value: str) -> bool:
        """Check for potentially malicious content."""
        # SQL injection patterns
        if self.patterns['sql_injection'].search(value):
            return True
        
        # Basic XSS patterns
        if self.patterns['xss_basic'].search(value):
            return True
        
        # Command injection patterns
        if self.patterns['command_injection'].search(value):
            return True
        
        # AI-specific threats
        if self.patterns['prompt_injection'].search(value):
            logger.warning(f"Prompt injection attempt detected: {value[:100]}...")
            return True
        
        if self.patterns['jailbreak_attempt'].search(value):
            logger.warning(f"Jailbreak attempt detected: {value[:100]}...")
            return True
        
        if self.patterns['model_extraction'].search(value):
            logger.warning(f"Model extraction attempt detected: {value[:100]}...")
            return True
        
        if self.patterns['adversarial_prompt'].search(value):
            logger.warning(f"Adversarial prompt detected: {value[:100]}...")
            return True
        
        # Advanced AI threat detection
        if self.ai_threat_detector.detect_advanced_threats(value):
            return True
        
        # Check for null bytes
        if '\x00' in value:
            return True
        
        return False
    
    def validate_multimodal_input(self, input_data: Dict[str, Any], client_ip: str = None) -> Dict[str, Any]:
        """Validate multimodal input with AI-specific security checks."""
        # Rate limiting check
        if client_ip and not self._check_rate_limit(client_ip):
            raise ValidationError("Rate limit exceeded for multimodal requests")
        
        # Validate multimodal content
        if 'type' in input_data and 'data' in input_data:
            input_type = input_data['type']
            input_content = input_data['data']
            
            # Use specialized multimodal validator
            if not self.multimodal_validator.validate_content(input_type, input_content):
                raise ValidationError(f"Invalid or potentially malicious {input_type} content")
        
        return input_data
    
    def _check_rate_limit(self, client_ip: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """Check if client has exceeded rate limit for multimodal requests."""
        current_time = datetime.utcnow()
        window_start = current_time - timedelta(minutes=window_minutes)
        
        # Initialize or clean up old entries
        if client_ip not in self.request_rate_tracker:
            self.request_rate_tracker[client_ip] = []
        
        # Remove old requests outside the window
        self.request_rate_tracker[client_ip] = [
            req_time for req_time in self.request_rate_tracker[client_ip] 
            if req_time > window_start
        ]
        
        # Check if limit exceeded
        if len(self.request_rate_tracker[client_ip]) >= max_requests:
            return False
        
        # Add current request
        self.request_rate_tracker[client_ip].append(current_time)
        return True


class InputSanitizer:
    """Input sanitization utilities."""
    
    def __init__(self):
        """Initialize input sanitizer."""
        # Bleach configuration for HTML sanitization
        self.allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'
        ]
        self.allowed_attributes = {
            '*': ['class'],
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'width', 'height']
        }
    
    def sanitize_string(self, value: str, max_length: int = None) -> str:
        """Sanitize a string value."""
        if not isinstance(value, str):
            value = str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Normalize whitespace
        value = ' '.join(value.split())
        
        # Truncate if needed
        if max_length and len(value) > max_length:
            value = value[:max_length]
        
        return value
    
    def sanitize_html(self, value: str) -> str:
        """Sanitize HTML content."""
        if not isinstance(value, str):
            value = str(value)
        
        # Use bleach to sanitize HTML
        sanitized = bleach.clean(
            value,
            tags=self.allowed_tags,
            attributes=self.allowed_attributes,
            strip=True
        )
        
        return sanitized
    
    def escape_html(self, value: str) -> str:
        """Escape HTML entities."""
        if not isinstance(value, str):
            value = str(value)
        
        return html.escape(value, quote=True)
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe filesystem usage."""
        if not isinstance(filename, str):
            filename = str(filename)
        
        # Remove or replace dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Prevent reserved names on Windows
        reserved_names = [
            'CON', 'PRN', 'AUX', 'NUL',
            'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
            'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
        ]
        
        if filename.upper() in reserved_names:
            filename = f"_{filename}"
        
        # Limit length
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
            max_name_length = 255 - len(ext) - 1 if ext else 255
            filename = f"{name[:max_name_length]}.{ext}" if ext else name[:255]
        
        return filename
    
    def sanitize_sql_identifier(self, identifier: str) -> str:
        """Sanitize SQL identifier (table/column name)."""
        if not isinstance(identifier, str):
            identifier = str(identifier)
        
        # Only allow alphanumeric and underscore
        identifier = re.sub(r'[^a-zA-Z0-9_]', '', identifier)
        
        # Ensure it starts with a letter or underscore
        if identifier and identifier[0].isdigit():
            identifier = f"_{identifier}"
        
        return identifier
    
    def sanitize_json(self, data: Any, max_depth: int = 10) -> Any:
        """Sanitize JSON data recursively."""
        return self._sanitize_json_recursive(data, max_depth, 0)
    
    def _sanitize_json_recursive(self, data: Any, max_depth: int, current_depth: int) -> Any:
        """Recursively sanitize JSON data."""
        if current_depth > max_depth:
            return None
        
        if isinstance(data, str):
            return self.sanitize_string(data, max_length=10000)
        elif isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if isinstance(key, str) and len(key) <= 100:  # Limit key length
                    clean_key = self.sanitize_string(key, max_length=100)
                    sanitized[clean_key] = self._sanitize_json_recursive(
                        value, max_depth, current_depth + 1
                    )
            return sanitized
        elif isinstance(data, list):
            if len(data) > 1000:  # Limit array size
                data = data[:1000]
            return [
                self._sanitize_json_recursive(item, max_depth, current_depth + 1)
                for item in data
            ]
        elif isinstance(data, (int, float, bool)) or data is None:
            return data
        else:
            # Convert unknown types to string
            return self.sanitize_string(str(data), max_length=1000)


# Global instances
_input_validator: Optional[InputValidator] = None
_input_sanitizer: Optional[InputSanitizer] = None


def get_input_validator() -> InputValidator:
    """Get global input validator instance."""
    global _input_validator
    
    if _input_validator is None:
        _input_validator = InputValidator()
        _setup_default_schemas()
    
    return _input_validator


def get_input_sanitizer() -> InputSanitizer:
    """Get global input sanitizer instance."""
    global _input_sanitizer
    
    if _input_sanitizer is None:
        _input_sanitizer = InputSanitizer()
    
    return _input_sanitizer


def _setup_default_schemas() -> None:
    """Setup default validation schemas."""
    validator = get_input_validator()
    
    # User registration schema
    validator.add_schema("user_registration", [
        ValidationRule("username", ValidationType.STRING, required=True, min_length=3, max_length=50, pattern=r'^[a-zA-Z0-9_]+$'),
        ValidationRule("email", ValidationType.EMAIL, required=True),
        ValidationRule("password", ValidationType.STRING, required=True, min_length=8, max_length=128)
    ])
    
    # Scenario creation schema
    validator.add_schema("scenario_creation", [
        ValidationRule("title", ValidationType.STRING, required=True, min_length=1, max_length=200),
        ValidationRule("description", ValidationType.STRING, required=True, min_length=10, max_length=2000),
        ValidationRule("category", ValidationType.STRING, required=True, allowed_values=["misinformation", "fraud", "manipulation", "pseudoscience"]),
        ValidationRule("correct_skepticism_level", ValidationType.FLOAT, required=True, min_value=0.0, max_value=10.0),
        ValidationRule("red_flags", ValidationType.LIST, required=False, max_length=20),
        ValidationRule("metadata", ValidationType.DICT, required=False)
    ])
    
    # API request schema
    validator.add_schema("api_request", [
        ValidationRule("model", ValidationType.STRING, required=True, max_length=100),
        ValidationRule("agent_provider", ValidationType.STRING, required=True, max_length=50),
        ValidationRule("scenario_ids", ValidationType.LIST, required=False, max_length=100)
    ])
    
    logger.info("Default validation schemas set up")


class AIThreatDetector:
    """Advanced AI-specific threat detection using statistical analysis."""
    
    def __init__(self):
        """Initialize AI threat detector."""
        self.suspicious_patterns = [
            # Encoding-based attempts to hide instructions
            r'(?:base64|hex|rot13|caesar)',
            # Unusual formatting that might confuse tokenizers
            r'[^\w\s]{10,}',
            # Repeated characters (potential overflow attempts)
            r'(.)\1{50,}',
            # Unusual unicode patterns
            r'[\u200b-\u200f\ufeff\u202a-\u202e]',
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.suspicious_patterns]
        
        # Track statistical anomalies
        self.baseline_entropy = 4.5  # Expected entropy for normal text
        self.baseline_token_ratio = 0.75  # Expected token/character ratio
    
    def detect_advanced_threats(self, text: str) -> bool:
        """Detect advanced AI threats using multiple techniques."""
        if not text or not isinstance(text, str):
            return False
        
        # Check for suspicious patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                logger.warning(f"Suspicious pattern detected in input")
                return True
        
        # Statistical analysis
        if self._analyze_text_statistics(text):
            return True
        
        # Check for hidden instructions
        if self._detect_hidden_instructions(text):
            return True
        
        # Check for token manipulation attempts
        if self._detect_token_manipulation(text):
            return True
        
        return False
    
    def _analyze_text_statistics(self, text: str) -> bool:
        """Analyze text statistics for anomalies."""
        try:
            # Calculate entropy
            entropy = self._calculate_entropy(text)
            if entropy > self.baseline_entropy * 1.5 or entropy < self.baseline_entropy * 0.3:
                logger.warning(f"Anomalous text entropy detected: {entropy:.2f}")
                return True
            
            # Check character distribution
            if self._has_unusual_char_distribution(text):
                return True
            
            # Check for excessive repetition
            if self._has_excessive_repetition(text):
                return True
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return False
        
        return False
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        text_length = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / text_length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _has_unusual_char_distribution(self, text: str) -> bool:
        """Check for unusual character distribution patterns."""
        # Count different character types
        alphanumeric = sum(1 for c in text if c.isalnum())
        punctuation = sum(1 for c in text if not c.isalnum() and not c.isspace())
        whitespace = sum(1 for c in text if c.isspace())
        
        total_chars = len(text)
        if total_chars == 0:
            return False
        
        # Check ratios
        punct_ratio = punctuation / total_chars
        alpha_ratio = alphanumeric / total_chars
        
        # Flag if unusual ratios
        if punct_ratio > 0.3 or alpha_ratio < 0.3:
            logger.warning(f"Unusual character distribution: punct={punct_ratio:.2f}, alpha={alpha_ratio:.2f}")
            return True
        
        return False
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Check for excessive character or substring repetition."""
        if len(text) < 10:
            return False
        
        # Check for repeated substrings
        for length in [2, 3, 4, 5]:
            substrings = {}
            for i in range(len(text) - length + 1):
                substring = text[i:i + length]
                substrings[substring] = substrings.get(substring, 0) + 1
            
            # Check if any substring appears too frequently
            for count in substrings.values():
                if count > len(text) // (length * 2):  # More than 50% coverage
                    logger.warning(f"Excessive repetition detected")
                    return True
        
        return False
    
    def _detect_hidden_instructions(self, text: str) -> bool:
        """Detect attempts to hide instructions in text."""
        # Check for base64 encoded content
        try:
            if len(text) > 20 and len(text) % 4 == 0:
                decoded = base64.b64decode(text, validate=True).decode('utf-8')
                # Check if decoded content contains instruction-like patterns
                instruction_patterns = ['system:', 'user:', 'assistant:', 'ignore', 'forget', 'act as']
                if any(pattern in decoded.lower() for pattern in instruction_patterns):
                    logger.warning("Hidden instructions detected in base64 content")
                    return True
        except Exception:
            pass
        
        # Check for reversed text
        reversed_text = text[::-1].lower()
        if any(pattern in reversed_text for pattern in ['metsys', 'resu', 'tnatsissa']):
            logger.warning("Reversed instruction patterns detected")
            return True
        
        return False
    
    def _detect_token_manipulation(self, text: str) -> bool:
        """Detect attempts to manipulate tokenization."""
        # Check for excessive use of special characters
        special_chars = sum(1 for c in text if ord(c) > 127 or c in '\\/{}"[]')
        if len(text) > 0 and special_chars / len(text) > 0.2:
            logger.warning("Excessive special character usage detected")
            return True
        
        # Check for unusual spacing patterns
        if re.search(r'\s{10,}|[^\s]{100,}', text):
            logger.warning("Unusual spacing pattern detected")
            return True
        
        return False


class MultiModalValidator:
    """Validator for multimodal input content."""
    
    def __init__(self):
        """Initialize multimodal validator."""
        self.max_image_size = 10 * 1024 * 1024  # 10MB
        self.max_audio_size = 50 * 1024 * 1024  # 50MB
        self.max_video_size = 100 * 1024 * 1024  # 100MB
        self.max_document_size = 5 * 1024 * 1024  # 5MB
        
        self.allowed_image_formats = ['JPEG', 'PNG', 'GIF', 'WEBP']
        self.allowed_audio_formats = ['mp3', 'wav', 'ogg', 'flac']
        self.allowed_video_formats = ['mp4', 'avi', 'mkv', 'webm']
        self.allowed_document_formats = ['pdf', 'txt', 'docx', 'rtf']
    
    def validate_content(self, content_type: str, content_data: Union[str, bytes]) -> bool:
        """Validate multimodal content based on type."""
        try:
            if content_type == 'image':
                return self._validate_image(content_data)
            elif content_type == 'audio':
                return self._validate_audio(content_data)
            elif content_type == 'video':
                return self._validate_video(content_data)
            elif content_type == 'document':
                return self._validate_document(content_data)
            else:
                logger.warning(f"Unknown content type: {content_type}")
                return False
        except Exception as e:
            logger.error(f"Error validating {content_type} content: {e}")
            return False
    
    def _validate_image(self, image_data: Union[str, bytes]) -> bool:
        """Validate image content."""
        try:
            # Convert from base64 if needed
            if isinstance(image_data, str):
                image_bytes = base64.b64decode(image_data)
            else:
                image_bytes = image_data
            
            # Size check
            if len(image_bytes) > self.max_image_size:
                logger.warning("Image size exceeds maximum allowed")
                return False
            
            # Basic format validation
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(image_bytes))
            
            # Check format
            if image.format not in self.allowed_image_formats:
                logger.warning(f"Unsupported image format: {image.format}")
                return False
            
            # Check dimensions (prevent extremely large images)
            width, height = image.size
            if width * height > 100_000_000:  # 100 megapixels max
                logger.warning("Image dimensions too large")
                return False
            
            # Check for potential steganography (basic)
            if self._check_steganography_indicators(image_bytes):
                logger.warning("Potential steganography detected")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Image validation error: {e}")
            return False
    
    def _validate_audio(self, audio_data: Union[str, bytes]) -> bool:
        """Validate audio content."""
        try:
            # Convert from base64 if needed
            if isinstance(audio_data, str):
                audio_bytes = base64.b64decode(audio_data)
            else:
                audio_bytes = audio_data
            
            # Size check
            if len(audio_bytes) > self.max_audio_size:
                logger.warning("Audio size exceeds maximum allowed")
                return False
            
            # Basic header validation for common formats
            if not self._validate_audio_header(audio_bytes):
                logger.warning("Invalid audio file header")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Audio validation error: {e}")
            return False
    
    def _validate_video(self, video_data: Union[str, bytes]) -> bool:
        """Validate video content."""
        try:
            # Convert from base64 if needed
            if isinstance(video_data, str):
                video_bytes = base64.b64decode(video_data)
            else:
                video_bytes = video_data
            
            # Size check
            if len(video_bytes) > self.max_video_size:
                logger.warning("Video size exceeds maximum allowed")
                return False
            
            # Basic header validation
            if not self._validate_video_header(video_bytes):
                logger.warning("Invalid video file header")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Video validation error: {e}")
            return False
    
    def _validate_document(self, document_data: Union[str, bytes]) -> bool:
        """Validate document content."""
        try:
            # Convert from base64 if needed
            if isinstance(document_data, str):
                document_bytes = base64.b64decode(document_data)
            else:
                document_bytes = document_data
            
            # Size check
            if len(document_bytes) > self.max_document_size:
                logger.warning("Document size exceeds maximum allowed")
                return False
            
            # Check for executable content
            if self._contains_executable_content(document_bytes):
                logger.warning("Document contains executable content")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Document validation error: {e}")
            return False
    
    def _check_steganography_indicators(self, image_bytes: bytes) -> bool:
        """Check for basic steganography indicators."""
        # Look for unusual patterns in least significant bits
        # This is a basic check - production would use more sophisticated methods
        
        # Check for high entropy in LSBs
        lsb_data = []
        for i in range(0, min(len(image_bytes), 10000), 4):  # Sample every 4th byte
            if i < len(image_bytes):
                lsb_data.append(image_bytes[i] & 1)  # Extract LSB
        
        if len(lsb_data) > 100:
            # Calculate entropy of LSBs
            ones = sum(lsb_data)
            zeros = len(lsb_data) - ones
            
            # Perfect randomness would be close to 50/50
            if ones > 0 and zeros > 0:
                ratio = min(ones, zeros) / max(ones, zeros)
                if ratio > 0.9:  # Very close to random
                    return True
        
        return False
    
    def _validate_audio_header(self, audio_bytes: bytes) -> bool:
        """Validate audio file headers."""
        if len(audio_bytes) < 12:
            return False
        
        # Check for common audio file signatures
        audio_signatures = [
            b'ID3',      # MP3 with ID3 tag
            b'\xff\xfb', # MP3 without ID3
            b'RIFF',     # WAV
            b'OggS',     # OGG
            b'fLaC',     # FLAC
        ]
        
        for signature in audio_signatures:
            if audio_bytes.startswith(signature) or signature in audio_bytes[:12]:
                return True
        
        return False
    
    def _validate_video_header(self, video_bytes: bytes) -> bool:
        """Validate video file headers."""
        if len(video_bytes) < 12:
            return False
        
        # Check for common video file signatures
        video_signatures = [
            b'ftyp',     # MP4/MOV
            b'RIFF',     # AVI
            b'\x1a\x45\xdf\xa3',  # MKV
            b'webm',     # WEBM
        ]
        
        for signature in video_signatures:
            if signature in video_bytes[:20]:
                return True
        
        return False
    
    def _contains_executable_content(self, document_bytes: bytes) -> bool:
        """Check for executable content in documents."""
        # Check for common executable signatures
        executable_signatures = [
            b'MZ',       # Windows executable
            b'\x7fELF',  # Linux executable
            b'PK\x03\x04',  # ZIP (potential macro container)
        ]
        
        for signature in executable_signatures:
            if document_bytes.startswith(signature):
                return True
        
        # Check for script content
        text_content = document_bytes[:1000].decode('utf-8', errors='ignore').lower()
        script_patterns = ['<script', 'javascript:', 'vbscript:', 'powershell', 'cmd.exe']
        
        for pattern in script_patterns:
            if pattern in text_content:
                return True
        
        return False