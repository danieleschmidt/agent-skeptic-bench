"""External service integrations for Agent Skeptic Bench."""

from .github import GitHubIntegration
from .notifications import NotificationManager, EmailNotifier, SlackNotifier
from .auth import AuthManager, OAuthHandler, JWTManager

__all__ = [
    "GitHubIntegration",
    "NotificationManager",
    "EmailNotifier", 
    "SlackNotifier",
    "AuthManager",
    "OAuthHandler",
    "JWTManager"
]