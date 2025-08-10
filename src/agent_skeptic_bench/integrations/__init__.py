"""External service integrations for Agent Skeptic Bench."""

from .auth import AuthManager, JWTManager, OAuthHandler
from .github import GitHubIntegration
from .notifications import EmailNotifier, NotificationManager, SlackNotifier

__all__ = [
    "GitHubIntegration",
    "NotificationManager",
    "EmailNotifier",
    "SlackNotifier",
    "AuthManager",
    "OAuthHandler",
    "JWTManager"
]
