"""Middleware for Agent Skeptic Bench API."""

import logging
import time
from collections.abc import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..database.connection import get_database
from ..database.repositories import AuditRepository

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        # Process request
        response = await call_next(request)

        # Calculate response time
        process_time = time.time() - start_time

        # Log response
        logger.info(
            f"Response: {response.status_code} "
            f"in {process_time:.3f}s"
        )

        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(self, app: FastAPI, calls_per_minute: int = 60):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.request_times = {}  # client_ip -> list of request times

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        # Clean old requests (older than 1 minute)
        if client_ip in self.request_times:
            self.request_times[client_ip] = [
                req_time for req_time in self.request_times[client_ip]
                if current_time - req_time < 60
            ]
        else:
            self.request_times[client_ip] = []

        # Check rate limit
        if len(self.request_times[client_ip]) >= self.calls_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.calls_per_minute} requests per minute allowed",
                    "type": "rate_limit_error"
                }
            )

        # Add current request time
        self.request_times[client_ip].append(current_time)

        # Process request
        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response


class AuditMiddleware(BaseHTTPMiddleware):
    """Middleware for audit logging."""

    def __init__(self, app: FastAPI, log_read_operations: bool = False):
        super().__init__(app)
        self.log_read_operations = log_read_operations

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip audit logging for health checks and docs
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Skip GET requests unless configured to log them
        if request.method == "GET" and not self.log_read_operations:
            return await call_next(request)

        # Process request
        response = await call_next(request)

        # Log to audit trail (async, don't block response)
        try:
            await self._log_audit_entry(request, response)
        except Exception as e:
            logger.warning(f"Failed to create audit log entry: {e}")

        return response

    async def _log_audit_entry(self, request: Request, response: Response) -> None:
        """Create audit log entry."""
        try:
            # Extract user ID from token (if available)
            user_id = None
            auth_header = request.headers.get("Authorization")
            if auth_header:
                # TODO: Extract user ID from JWT token
                user_id = "authenticated_user"

            # Determine resource type and ID from path
            path_parts = request.url.path.strip("/").split("/")
            resource_type = "unknown"
            resource_id = None

            if len(path_parts) >= 3 and path_parts[0] == "api" and path_parts[1] == "v1":
                resource_type = path_parts[2]
                if len(path_parts) >= 4 and path_parts[3] != "":
                    resource_id = path_parts[3]

            # Map HTTP methods to actions
            action_map = {
                "GET": "read",
                "POST": "create",
                "PUT": "update",
                "PATCH": "update",
                "DELETE": "delete"
            }
            action = action_map.get(request.method, request.method.lower())

            # Create audit log entry
            db = get_database()
            async with db.get_async_session() as session:
                audit_repo = AuditRepository(session)
                await audit_repo.log(
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    user_id=user_id,
                    details={
                        "method": request.method,
                        "path": str(request.url.path),
                        "status_code": response.status_code,
                        "query_params": dict(request.query_params),
                        "user_agent": request.headers.get("User-Agent"),
                    },
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("User-Agent")
                )
                await audit_repo.commit()

        except Exception as e:
            logger.error(f"Error creating audit log entry: {e}")


def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the application."""

    # Security headers (first)
    app.add_middleware(SecurityHeadersMiddleware)

    # Rate limiting
    import os
    rate_limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    app.add_middleware(RateLimitMiddleware, calls_per_minute=rate_limit)

    # Request logging
    app.add_middleware(RequestLoggingMiddleware)

    # Audit logging (last, so it sees the final response)
    log_reads = os.getenv("AUDIT_LOG_READ_OPERATIONS", "false").lower() == "true"
    app.add_middleware(AuditMiddleware, log_read_operations=log_reads)

    logger.info("API middleware setup complete")
