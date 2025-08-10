"""Rate limiting for Agent Skeptic Bench."""

import asyncio
import logging
import threading
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimitScope(Enum):
    """Rate limit scopes."""

    GLOBAL = "global"
    USER = "user"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    name: str
    strategy: RateLimitStrategy
    scope: RateLimitScope
    limit: int
    window_seconds: int
    burst_limit: int | None = None
    retry_after: int | None = None
    enabled: bool = True
    whitelist: list[str] = None
    blacklist: list[str] = None


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: int | None = None
    message: str = ""


@dataclass
class RateLimitViolation:
    """Rate limit violation record."""

    config_name: str
    identifier: str
    scope: RateLimitScope
    timestamp: datetime
    current_count: int
    limit: int
    duration: int


class RateLimiter:
    """Comprehensive rate limiting system."""

    def __init__(self):
        """Initialize rate limiter."""
        self.configs: dict[str, RateLimitConfig] = {}
        self.counters: dict[str, Any] = defaultdict(dict)
        self.violations: list[RateLimitViolation] = []
        self._lock = threading.Lock()

        # Strategy implementations
        self.strategies = {
            RateLimitStrategy.FIXED_WINDOW: self._fixed_window_check,
            RateLimitStrategy.SLIDING_WINDOW: self._sliding_window_check,
            RateLimitStrategy.TOKEN_BUCKET: self._token_bucket_check,
            RateLimitStrategy.LEAKY_BUCKET: self._leaky_bucket_check
        }

    def add_config(self, config: RateLimitConfig) -> None:
        """Add a rate limit configuration."""
        self.configs[config.name] = config
        logger.info(f"Added rate limit config: {config.name}")

    def remove_config(self, name: str) -> None:
        """Remove a rate limit configuration."""
        if name in self.configs:
            del self.configs[name]
            # Clean up counters for this config
            with self._lock:
                if name in self.counters:
                    del self.counters[name]
            logger.info(f"Removed rate limit config: {name}")

    async def check_rate_limit(self, config_name: str, identifier: str,
                             request_count: int = 1) -> RateLimitResult:
        """Check if request is within rate limits."""
        config = self.configs.get(config_name)
        if not config or not config.enabled:
            return RateLimitResult(
                allowed=True,
                remaining=float('inf'),
                reset_time=datetime.utcnow() + timedelta(seconds=config.window_seconds if config else 3600)
            )

        # Check whitelist/blacklist
        if config.whitelist and identifier in config.whitelist:
            return RateLimitResult(
                allowed=True,
                remaining=float('inf'),
                reset_time=datetime.utcnow() + timedelta(seconds=config.window_seconds)
            )

        if config.blacklist and identifier in config.blacklist:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=datetime.utcnow() + timedelta(seconds=config.window_seconds),
                message="Identifier is blacklisted"
            )

        # Apply rate limiting strategy
        strategy_func = self.strategies.get(config.strategy)
        if not strategy_func:
            logger.warning(f"Unknown rate limit strategy: {config.strategy}")
            return RateLimitResult(
                allowed=True,
                remaining=config.limit,
                reset_time=datetime.utcnow() + timedelta(seconds=config.window_seconds)
            )

        result = await strategy_func(config, identifier, request_count)

        # Record violation if rate limit exceeded
        if not result.allowed:
            await self._record_violation(config, identifier, result)

        return result

    async def _fixed_window_check(self, config: RateLimitConfig, identifier: str,
                                request_count: int) -> RateLimitResult:
        """Fixed window rate limiting."""
        now = datetime.utcnow()
        window_start = now.replace(second=0, microsecond=0)

        # Calculate which window we're in
        window_number = int(window_start.timestamp()) // config.window_seconds

        with self._lock:
            if config.name not in self.counters:
                self.counters[config.name] = {}

            config_counters = self.counters[config.name]
            key = f"{identifier}:{window_number}"

            current_count = config_counters.get(key, 0)

            if current_count + request_count <= config.limit:
                config_counters[key] = current_count + request_count
                remaining = config.limit - (current_count + request_count)
                allowed = True
            else:
                remaining = 0
                allowed = False

            # Clean up old windows
            self._cleanup_old_windows(config_counters, window_number, identifier)

            # Calculate reset time
            reset_time = datetime.fromtimestamp((window_number + 1) * config.window_seconds)

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=config.retry_after if not allowed else None
            )

    async def _sliding_window_check(self, config: RateLimitConfig, identifier: str,
                                  request_count: int) -> RateLimitResult:
        """Sliding window rate limiting."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=config.window_seconds)

        with self._lock:
            if config.name not in self.counters:
                self.counters[config.name] = {}

            config_counters = self.counters[config.name]

            if identifier not in config_counters:
                config_counters[identifier] = deque()

            request_times = config_counters[identifier]

            # Remove requests outside the window
            while request_times and request_times[0] < window_start:
                request_times.popleft()

            current_count = len(request_times)

            if current_count + request_count <= config.limit:
                # Add new requests
                for _ in range(request_count):
                    request_times.append(now)
                remaining = config.limit - (current_count + request_count)
                allowed = True
            else:
                remaining = 0
                allowed = False

            # Calculate reset time (when oldest request in window expires)
            if request_times:
                reset_time = request_times[0] + timedelta(seconds=config.window_seconds)
            else:
                reset_time = now + timedelta(seconds=config.window_seconds)

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=config.retry_after if not allowed else None
            )

    async def _token_bucket_check(self, config: RateLimitConfig, identifier: str,
                                request_count: int) -> RateLimitResult:
        """Token bucket rate limiting."""
        now = datetime.utcnow()

        with self._lock:
            if config.name not in self.counters:
                self.counters[config.name] = {}

            config_counters = self.counters[config.name]

            if identifier not in config_counters:
                config_counters[identifier] = {
                    'tokens': config.limit,
                    'last_refill': now
                }

            bucket = config_counters[identifier]

            # Calculate tokens to add based on time elapsed
            time_elapsed = (now - bucket['last_refill']).total_seconds()
            tokens_to_add = int(time_elapsed * (config.limit / config.window_seconds))

            # Refill bucket (up to limit)
            bucket['tokens'] = min(config.limit, bucket['tokens'] + tokens_to_add)
            bucket['last_refill'] = now

            if bucket['tokens'] >= request_count:
                bucket['tokens'] -= request_count
                remaining = bucket['tokens']
                allowed = True
            else:
                remaining = bucket['tokens']
                allowed = False

            # Calculate when bucket will have enough tokens
            if not allowed and request_count > 0:
                time_to_refill = (request_count - bucket['tokens']) / (config.limit / config.window_seconds)
                reset_time = now + timedelta(seconds=time_to_refill)
            else:
                reset_time = now + timedelta(seconds=config.window_seconds)

            return RateLimitResult(
                allowed=allowed,
                remaining=remaining,
                reset_time=reset_time,
                retry_after=int(time_to_refill) if not allowed else None
            )

    async def _leaky_bucket_check(self, config: RateLimitConfig, identifier: str,
                                request_count: int) -> RateLimitResult:
        """Leaky bucket rate limiting."""
        now = datetime.utcnow()

        with self._lock:
            if config.name not in self.counters:
                self.counters[config.name] = {}

            config_counters = self.counters[config.name]

            if identifier not in config_counters:
                config_counters[identifier] = {
                    'level': 0,
                    'last_leak': now
                }

            bucket = config_counters[identifier]

            # Calculate leak amount based on time elapsed
            time_elapsed = (now - bucket['last_leak']).total_seconds()
            leak_amount = time_elapsed * (config.limit / config.window_seconds)

            # Apply leak
            bucket['level'] = max(0, bucket['level'] - leak_amount)
            bucket['last_leak'] = now

            # Check if we can add the request
            bucket_capacity = config.burst_limit or config.limit

            if bucket['level'] + request_count <= bucket_capacity:
                bucket['level'] += request_count
                remaining = bucket_capacity - bucket['level']
                allowed = True
            else:
                remaining = bucket_capacity - bucket['level']
                allowed = False

            # Calculate when bucket will have space
            if not allowed:
                time_to_space = (bucket['level'] + request_count - bucket_capacity) / (config.limit / config.window_seconds)
                reset_time = now + timedelta(seconds=time_to_space)
            else:
                reset_time = now + timedelta(seconds=config.window_seconds)

            return RateLimitResult(
                allowed=allowed,
                remaining=int(remaining),
                reset_time=reset_time,
                retry_after=int(time_to_space) if not allowed else None
            )

    def _cleanup_old_windows(self, config_counters: dict[str, int],
                           current_window: int, identifier: str) -> None:
        """Clean up old window data."""
        keys_to_remove = []

        for key in config_counters.keys():
            if key.startswith(f"{identifier}:"):
                try:
                    window_num = int(key.split(":")[1])
                    if window_num < current_window - 1:  # Keep current and previous window
                        keys_to_remove.append(key)
                except (IndexError, ValueError):
                    pass

        for key in keys_to_remove:
            del config_counters[key]

    async def _record_violation(self, config: RateLimitConfig, identifier: str,
                              result: RateLimitResult) -> None:
        """Record a rate limit violation."""
        violation = RateLimitViolation(
            config_name=config.name,
            identifier=identifier,
            scope=config.scope,
            timestamp=datetime.utcnow(),
            current_count=config.limit - result.remaining + 1,
            limit=config.limit,
            duration=config.window_seconds
        )

        self.violations.append(violation)

        # Keep only recent violations (last 1000)
        if len(self.violations) > 1000:
            self.violations = self.violations[-1000:]

        logger.warning(f"Rate limit violation: {config.name} for {identifier}")

    def get_rate_limit_status(self, config_name: str, identifier: str) -> dict[str, Any]:
        """Get current rate limit status for an identifier."""
        config = self.configs.get(config_name)
        if not config:
            return {"error": "Configuration not found"}

        with self._lock:
            if config.name not in self.counters or identifier not in self.counters[config.name]:
                return {
                    "remaining": config.limit,
                    "reset_time": datetime.utcnow() + timedelta(seconds=config.window_seconds),
                    "current_usage": 0
                }

            # Get current state based on strategy
            if config.strategy == RateLimitStrategy.SLIDING_WINDOW:
                now = datetime.utcnow()
                window_start = now - timedelta(seconds=config.window_seconds)
                request_times = self.counters[config.name][identifier]

                # Count requests in window
                current_usage = sum(1 for req_time in request_times if req_time > window_start)
                remaining = config.limit - current_usage
                reset_time = request_times[0] + timedelta(seconds=config.window_seconds) if request_times else now

            elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
                bucket = self.counters[config.name][identifier]
                remaining = bucket['tokens']
                current_usage = config.limit - remaining
                reset_time = bucket['last_refill'] + timedelta(seconds=config.window_seconds)

            else:  # Fixed window and others
                current_usage = 0
                remaining = config.limit
                reset_time = datetime.utcnow() + timedelta(seconds=config.window_seconds)

            return {
                "remaining": remaining,
                "reset_time": reset_time.isoformat(),
                "current_usage": current_usage,
                "limit": config.limit,
                "window_seconds": config.window_seconds
            }

    def get_violations(self, hours: int = 24) -> list[RateLimitViolation]:
        """Get recent rate limit violations."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [v for v in self.violations if v.timestamp > cutoff]

    def get_statistics(self) -> dict[str, Any]:
        """Get rate limiting statistics."""
        with self._lock:
            total_configs = len(self.configs)
            active_configs = len([c for c in self.configs.values() if c.enabled])

            # Count active rate limit states
            active_limiters = 0
            for config_counters in self.counters.values():
                active_limiters += len(config_counters)

            # Recent violations
            recent_violations = self.get_violations(hours=24)

            return {
                "total_configs": total_configs,
                "active_configs": active_configs,
                "active_limiters": active_limiters,
                "violations_24h": len(recent_violations),
                "violation_rate": len(recent_violations) / 24 if recent_violations else 0
            }


# Rate limiting decorator
def rate_limit(config_name: str, get_identifier: Callable[[Any], str] = None):
    """Decorator for rate limiting functions."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            limiter = get_rate_limiter()

            # Get identifier
            if get_identifier:
                identifier = get_identifier(args[0] if args else None)
            else:
                identifier = "default"

            # Check rate limit
            result = await limiter.check_rate_limit(config_name, identifier)

            if not result.allowed:
                raise RateLimitExceeded(
                    f"Rate limit exceeded for {config_name}",
                    retry_after=result.retry_after
                )

            return await func(*args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't easily do async rate limiting
            # This would require running in an async context
            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter

    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
        _setup_default_configs()

    return _rate_limiter


def _setup_default_configs() -> None:
    """Setup default rate limiting configurations."""
    limiter = get_rate_limiter()

    # API rate limiting
    limiter.add_config(RateLimitConfig(
        name="api_requests",
        strategy=RateLimitStrategy.SLIDING_WINDOW,
        scope=RateLimitScope.USER,
        limit=1000,
        window_seconds=3600,  # 1000 requests per hour
        retry_after=60
    ))

    # Evaluation rate limiting
    limiter.add_config(RateLimitConfig(
        name="evaluations",
        strategy=RateLimitStrategy.TOKEN_BUCKET,
        scope=RateLimitScope.USER,
        limit=100,
        window_seconds=3600,  # 100 evaluations per hour
        burst_limit=10,
        retry_after=120
    ))

    # Authentication rate limiting
    limiter.add_config(RateLimitConfig(
        name="auth_attempts",
        strategy=RateLimitStrategy.FIXED_WINDOW,
        scope=RateLimitScope.IP_ADDRESS,
        limit=10,
        window_seconds=900,  # 10 attempts per 15 minutes
        retry_after=900
    ))

    logger.info("Default rate limit configurations set up")
