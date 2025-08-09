"""Security module for Agent Skeptic Bench."""

# Handle optional security dependencies
try:
    from .authentication import AuthenticationManager, TokenManager, APIKeyManager
    auth_available = True
except ImportError:
    # Fallback stubs for missing security dependencies
    class AuthenticationManager:
        def __init__(self, secret_key=None): 
            self.secret_key = secret_key or "demo_secret"
        def authenticate(self, *args): return True
        def create_user(self, *args, **kwargs): return None
        def authenticate_user(self, *args): return None
    
    class TokenManager:
        def __init__(self, secret_key=None, issuer=None): 
            self.secret_key = secret_key or "demo_secret"
            self.issuer = issuer or "demo"
        def create_access_token(self, *args, **kwargs): return "demo_access_token"
        def create_refresh_token(self, *args, **kwargs): return "demo_refresh_token"
        def verify_token(self, *args): return None
    
    class APIKeyManager:
        def __init__(self): pass
        def create_api_key(self, *args, **kwargs): return ("demo_key", None)
        def verify_api_key(self, *args): return None
        def revoke_api_key(self, *args): return False
    
    auth_available = False

try:
    from .input_validation import InputValidator, InputSanitizer
    validation_available = True
except ImportError:
    # Fallback stubs for missing validation dependencies
    class InputValidator:
        def __init__(self): pass
        def add_schema(self, *args): pass
        def validate_data(self, schema_name, data): return data
        def validate_text(self, text): return text
    
    class InputSanitizer:
        def __init__(self): pass
        def sanitize_string(self, value, max_length=None): 
            if max_length and len(str(value)) > max_length:
                return str(value)[:max_length]
            return str(value)
        def sanitize_html(self, value): return str(value)
        def escape_html(self, value): return str(value)
        def sanitize_filename(self, filename): return str(filename).replace('/', '_')
        def sanitize_sql_identifier(self, identifier): return str(identifier)
        def sanitize_json(self, data, max_depth=10): return data
    
    validation_available = False

from .audit import AuditLogger, SecurityEvent, SecurityEventType
from .rate_limiting import RateLimiter, RateLimitConfig

__all__ = [
    "AuthenticationManager",
    "TokenManager",
    "APIKeyManager",
    "AuditLogger",
    "SecurityEvent",
    "SecurityEventType",
    "InputValidator",
    "InputSanitizer",
    "RateLimiter",
    "RateLimitConfig",
    "auth_available",
    "validation_available"
]