"""Input validation and sanitization for Agent Skeptic Bench."""

import html
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Handle optional bleach dependency
try:
    import bleach
    BLEACH_AVAILABLE = True
except ImportError:
    # Fallback when bleach is not available
    BLEACH_AVAILABLE = False

from urllib.parse import urlparse

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
    min_length: int | None = None
    max_length: int | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    pattern: str | None = None
    allowed_values: list[Any] | None = None
    custom_validator: Callable[[Any], bool] | None = None
    description: str = ""


class InputValidator:
    """Comprehensive input validation system."""

    def __init__(self):
        """Initialize input validator."""
        self.validation_schemas: dict[str, list[ValidationRule]] = {}

        # Pre-compiled patterns
        self.patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', re.IGNORECASE),
            'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
            'safe_string': re.compile(r'^[a-zA-Z0-9\s\-_.]+$'),
            'sql_injection': re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)', re.IGNORECASE),
            'xss_basic': re.compile(r'<\s*script[^>]*>.*?<\s*/\s*script\s*>', re.IGNORECASE | re.DOTALL),
            'command_injection': re.compile(r'[;&|`$\(\)]')
        }

    def add_schema(self, schema_name: str, rules: list[ValidationRule]) -> None:
        """Add a validation schema."""
        self.validation_schemas[schema_name] = rules
        logger.info(f"Added validation schema: {schema_name}")

    def validate_data(self, schema_name: str, data: dict[str, Any]) -> dict[str, Any]:
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

    def validate_text(self, text: str, max_length: int = 10000) -> str:
        """Validate and sanitize text input."""
        if not isinstance(text, str):
            try:
                text = str(text)
            except:
                raise ValidationError("Value must be a string")

        # Check length
        if len(text) > max_length:
            raise ValidationError(f"Text too long (max {max_length} characters)")

        # Basic security checks
        if self.patterns['sql_injection'].search(text):
            raise ValidationError("Potentially unsafe SQL patterns detected")

        if self.patterns['xss_basic'].search(text):
            raise ValidationError("Potentially unsafe script patterns detected")

        if self.patterns['command_injection'].search(text):
            raise ValidationError("Potentially unsafe command injection patterns detected")

        return text

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
                raise ValidationError("String does not match required pattern")

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

    def _validate_list(self, rule: ValidationRule, value: Any) -> list[Any]:
        """Validate list."""
        if not isinstance(value, list):
            raise ValidationError("Value must be a list")

        # Length validation
        if rule.min_length is not None and len(value) < rule.min_length:
            raise ValidationError(f"List must have at least {rule.min_length} items")

        if rule.max_length is not None and len(value) > rule.max_length:
            raise ValidationError(f"List must have at most {rule.max_length} items")

        return value

    def _validate_dict(self, rule: ValidationRule, value: Any) -> dict[str, Any]:
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

        # Check for null bytes
        if '\x00' in value:
            return True

        return False


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

        # Fallback HTML tag regex patterns when bleach is not available
        self._script_pattern = re.compile(r'<\s*script[^>]*>.*?<\s*/\s*script\s*>', re.IGNORECASE | re.DOTALL)
        self._tag_pattern = re.compile(r'<[^>]*>', re.IGNORECASE)

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

        if BLEACH_AVAILABLE:
            # Use bleach to sanitize HTML
            sanitized = bleach.clean(
                value,
                tags=self.allowed_tags,
                attributes=self.allowed_attributes,
                strip=True
            )
            return sanitized
        else:
            # Fallback: remove all HTML tags and escape entities
            logger.warning("bleach not available, using basic HTML sanitization fallback")

            # Remove script tags first
            value = self._script_pattern.sub('', value)

            # Remove all other HTML tags
            value = self._tag_pattern.sub('', value)

            # Escape any remaining HTML entities
            return html.escape(value, quote=True)

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
_input_validator: InputValidator | None = None
_input_sanitizer: InputSanitizer | None = None


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
