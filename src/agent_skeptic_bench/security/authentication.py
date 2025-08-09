"""Authentication and token management for Agent Skeptic Bench."""

import logging
import secrets
import hashlib
import hmac
import jwt
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
# Handle optional bcrypt dependency
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    # Fallback for when bcrypt is not available
    BCRYPT_AVAILABLE = False
    import hashlib
    
import base64


logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Authentication methods."""
    
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"


class TokenType(Enum):
    """Token types."""
    
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


@dataclass
class User:
    """User account representation."""
    
    id: str
    username: str
    email: str
    password_hash: str
    roles: List[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    metadata: Dict[str, Any] = None


@dataclass
class AuthToken:
    """Authentication token."""
    
    token: str
    token_type: TokenType
    user_id: str
    expires_at: datetime
    scopes: List[str]
    created_at: datetime
    revoked: bool = False


@dataclass
class APIKey:
    """API Key representation."""
    
    id: str
    name: str
    key_hash: str
    user_id: str
    scopes: List[str]
    expires_at: Optional[datetime]
    created_at: datetime
    last_used: Optional[datetime] = None
    revoked: bool = False


class AuthenticationManager:
    """Main authentication manager."""
    
    def __init__(self, secret_key: str):
        """Initialize authentication manager."""
        self.secret_key = secret_key
        self.users: Dict[str, User] = {}
        self.active_tokens: Dict[str, AuthToken] = {}
        self.failed_attempts: Dict[str, List[datetime]] = {}
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        self.password_policy = {
            'min_length': 8,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_numbers': True,
            'require_special': True
        }
    
    def create_user(self, username: str, email: str, password: str, 
                   roles: List[str] = None) -> User:
        """Create a new user account."""
        # Validate password
        if not self._validate_password(password):
            raise ValueError("Password does not meet security requirements")
        
        # Check if user already exists
        if any(u.username == username or u.email == email for u in self.users.values()):
            raise ValueError("User with this username or email already exists")
        
        # Generate user ID and hash password
        user_id = secrets.token_urlsafe(16)
        password_hash = self._hash_password(password)
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            roles=roles or ['user'],
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        self.users[user_id] = user
        logger.info(f"Created user account: {username}")
        
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        # Find user
        user = next((u for u in self.users.values() if u.username == username), None)
        if not user:
            self._record_failed_attempt(username)
            return None
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            logger.warning(f"Login attempt on locked account: {username}")
            return None
        
        # Check if account is active
        if not user.is_active:
            logger.warning(f"Login attempt on inactive account: {username}")
            return None
        
        # Verify password
        if not self._verify_password(password, user.password_hash):
            self._record_failed_attempt(username)
            user.failed_login_attempts += 1
            
            # Lock account if too many failed attempts
            if user.failed_login_attempts >= self.max_failed_attempts:
                user.locked_until = datetime.utcnow() + self.lockout_duration
                logger.warning(f"Account locked due to failed attempts: {username}")
            
            return None
        
        # Successful authentication
        user.last_login = datetime.utcnow()
        user.failed_login_attempts = 0
        user.locked_until = None
        
        # Clear failed attempts record
        if username in self.failed_attempts:
            del self.failed_attempts[username]
        
        logger.info(f"User authenticated successfully: {username}")
        return user
    
    def _validate_password(self, password: str) -> bool:
        """Validate password against security policy."""
        policy = self.password_policy
        
        if len(password) < policy['min_length']:
            return False
        
        if policy['require_uppercase'] and not any(c.isupper() for c in password):
            return False
        
        if policy['require_lowercase'] and not any(c.islower() for c in password):
            return False
        
        if policy['require_numbers'] and not any(c.isdigit() for c in password):
            return False
        
        if policy['require_special']:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(c in special_chars for c in password):
                return False
        
        return True
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt or fallback to SHA-256 with salt."""
        if BCRYPT_AVAILABLE:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        else:
            # Fallback to SHA-256 with random salt
            logger.warning("bcrypt not available, using SHA-256 fallback for password hashing")
            salt = secrets.token_hex(16)
            hashed = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
            return f"pbkdf2_sha256$100000${salt}${base64.b64encode(hashed).decode('utf-8')}"
    
    def _verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        if hashed.startswith('pbkdf2_sha256$'):
            # Handle fallback hash format: pbkdf2_sha256$iterations$salt$hash
            try:
                _, iterations_str, salt, stored_hash = hashed.split('$', 3)
                iterations = int(iterations_str)
                computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), iterations)
                return base64.b64encode(computed_hash).decode('utf-8') == stored_hash
            except (ValueError, TypeError):
                logger.error("Invalid fallback password hash format")
                return False
        elif BCRYPT_AVAILABLE:
            # Use bcrypt verification
            try:
                return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
            except (ValueError, TypeError):
                logger.error("Invalid bcrypt password hash")
                return False
        else:
            # bcrypt not available and it's a bcrypt hash - cannot verify
            logger.error("Cannot verify bcrypt hash when bcrypt is not available")
            return False
    
    def _record_failed_attempt(self, username: str) -> None:
        """Record failed login attempt."""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(datetime.utcnow())
        
        # Clean old attempts (keep only last hour)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.failed_attempts[username] = [
            attempt for attempt in self.failed_attempts[username]
            if attempt > cutoff
        ]
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> bool:
        """Change user password."""
        user = self.users.get(user_id)
        if not user:
            return False
        
        # Verify old password
        if not self._verify_password(old_password, user.password_hash):
            return False
        
        # Validate new password
        if not self._validate_password(new_password):
            raise ValueError("New password does not meet security requirements")
        
        # Update password
        user.password_hash = self._hash_password(new_password)
        logger.info(f"Password changed for user: {user.username}")
        
        return True
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user account."""
        user = self.users.get(user_id)
        if not user:
            return False
        
        user.is_active = False
        
        # Revoke all active tokens for this user
        for token in self.active_tokens.values():
            if token.user_id == user_id:
                token.revoked = True
        
        logger.info(f"User account deactivated: {user.username}")
        return True
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return next((u for u in self.users.values() if u.username == username), None)


class TokenManager:
    """JWT token management."""
    
    def __init__(self, secret_key: str, issuer: str = "agent-skeptic-bench"):
        """Initialize token manager."""
        self.secret_key = secret_key
        self.issuer = issuer
        self.algorithm = "HS256"
        self.access_token_ttl = timedelta(hours=1)
        self.refresh_token_ttl = timedelta(days=30)
        self.active_tokens: Dict[str, AuthToken] = {}
    
    def create_access_token(self, user: User, scopes: List[str] = None) -> str:
        """Create JWT access token."""
        now = datetime.utcnow()
        expires_at = now + self.access_token_ttl
        
        payload = {
            'sub': user.id,
            'username': user.username,
            'email': user.email,
            'roles': user.roles,
            'scopes': scopes or [],
            'iat': now,
            'exp': expires_at,
            'iss': self.issuer,
            'type': 'access'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Store token
        auth_token = AuthToken(
            token=token,
            token_type=TokenType.ACCESS,
            user_id=user.id,
            expires_at=expires_at,
            scopes=scopes or [],
            created_at=now
        )
        self.active_tokens[token] = auth_token
        
        return token
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        now = datetime.utcnow()
        expires_at = now + self.refresh_token_ttl
        
        payload = {
            'sub': user.id,
            'iat': now,
            'exp': expires_at,
            'iss': self.issuer,
            'type': 'refresh'
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Store token
        auth_token = AuthToken(
            token=token,
            token_type=TokenType.REFRESH,
            user_id=user.id,
            expires_at=expires_at,
            scopes=[],
            created_at=now
        )
        self.active_tokens[token] = auth_token
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token."""
        try:
            # Check if token is revoked
            auth_token = self.active_tokens.get(token)
            if auth_token and auth_token.revoked:
                return None
            
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is expired
            exp = payload.get('exp')
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Create new access token using refresh token."""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get('type') != 'refresh':
            return None
        
        # Get user
        user_id = payload.get('sub')
        # In a real implementation, you'd fetch the user from the auth manager
        # For now, we'll create a mock user
        
        # Create new access token
        from . import User  # Import here to avoid circular imports
        user = User(
            id=user_id,
            username="user",
            email="user@example.com",
            password_hash="",
            roles=["user"],
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        return self.create_access_token(user)
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a token."""
        auth_token = self.active_tokens.get(token)
        if auth_token:
            auth_token.revoked = True
            return True
        return False
    
    def revoke_user_tokens(self, user_id: str) -> int:
        """Revoke all tokens for a user."""
        count = 0
        for auth_token in self.active_tokens.values():
            if auth_token.user_id == user_id and not auth_token.revoked:
                auth_token.revoked = True
                count += 1
        return count
    
    def cleanup_expired_tokens(self) -> int:
        """Remove expired tokens from storage."""
        now = datetime.utcnow()
        expired_tokens = [
            token for token, auth_token in self.active_tokens.items()
            if auth_token.expires_at < now
        ]
        
        for token in expired_tokens:
            del self.active_tokens[token]
        
        return len(expired_tokens)


class APIKeyManager:
    """API key management."""
    
    def __init__(self):
        """Initialize API key manager."""
        self.api_keys: Dict[str, APIKey] = {}
    
    def create_api_key(self, user_id: str, name: str, scopes: List[str] = None,
                      expires_at: Optional[datetime] = None) -> Tuple[str, APIKey]:
        """Create a new API key."""
        # Generate API key
        key = self._generate_api_key()
        key_hash = self._hash_api_key(key)
        
        api_key = APIKey(
            id=secrets.token_urlsafe(16),
            name=name,
            key_hash=key_hash,
            user_id=user_id,
            scopes=scopes or [],
            expires_at=expires_at,
            created_at=datetime.utcnow()
        )
        
        self.api_keys[api_key.id] = api_key
        logger.info(f"Created API key: {name} for user: {user_id}")
        
        return key, api_key
    
    def verify_api_key(self, key: str) -> Optional[APIKey]:
        """Verify API key and return associated data."""
        key_hash = self._hash_api_key(key)
        
        for api_key in self.api_keys.values():
            if (api_key.key_hash == key_hash and 
                not api_key.revoked and
                (not api_key.expires_at or api_key.expires_at > datetime.utcnow())):
                
                # Update last used timestamp
                api_key.last_used = datetime.utcnow()
                return api_key
        
        return None
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        api_key = self.api_keys.get(key_id)
        if api_key:
            api_key.revoked = True
            logger.info(f"Revoked API key: {api_key.name}")
            return True
        return False
    
    def list_user_api_keys(self, user_id: str) -> List[APIKey]:
        """List all API keys for a user."""
        return [
            api_key for api_key in self.api_keys.values()
            if api_key.user_id == user_id and not api_key.revoked
        ]
    
    def _generate_api_key(self) -> str:
        """Generate a new API key."""
        # Format: asb_<random_string>
        random_part = secrets.token_urlsafe(32)
        return f"asb_{random_part}"
    
    def _hash_api_key(self, key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(key.encode()).hexdigest()


# Global instances
_auth_manager: Optional[AuthenticationManager] = None
_token_manager: Optional[TokenManager] = None
_api_key_manager: Optional[APIKeyManager] = None


def get_auth_manager() -> AuthenticationManager:
    """Get global authentication manager."""
    global _auth_manager
    
    if _auth_manager is None:
        secret_key = secrets.token_urlsafe(32)  # In production, use config
        _auth_manager = AuthenticationManager(secret_key)
    
    return _auth_manager


def get_token_manager() -> TokenManager:
    """Get global token manager."""
    global _token_manager
    
    if _token_manager is None:
        secret_key = secrets.token_urlsafe(32)  # In production, use config
        _token_manager = TokenManager(secret_key)
    
    return _token_manager


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager."""
    global _api_key_manager
    
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    
    return _api_key_manager