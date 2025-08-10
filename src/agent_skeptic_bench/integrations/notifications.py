"""Notification system for Agent Skeptic Bench."""

import asyncio
import logging
import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class NotificationMessage:
    """Represents a notification message."""

    title: str
    content: str
    level: str = "info"  # info, warning, error, success
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @abstractmethod
    async def send(self, message: NotificationMessage) -> bool:
        """Send a notification message."""
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the notification channel is available."""
        pass


class EmailNotifier(NotificationChannel):
    """Email notification channel."""

    def __init__(self,
                 smtp_host: str | None = None,
                 smtp_port: int | None = None,
                 smtp_user: str | None = None,
                 smtp_password: str | None = None,
                 from_email: str | None = None,
                 to_emails: list[str] | None = None):
        """Initialize email notifier.
        
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            smtp_user: SMTP username
            smtp_password: SMTP password
            from_email: From email address
            to_emails: List of recipient email addresses
        """
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "localhost")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("EMAIL_FROM", "noreply@skeptic-bench.org")
        self.to_emails = to_emails or self._parse_to_emails()

        self.use_tls = os.getenv("SMTP_TLS", "true").lower() == "true"

    def _parse_to_emails(self) -> list[str]:
        """Parse TO emails from environment variable."""
        to_emails_str = os.getenv("EMAIL_TO", "")
        if to_emails_str:
            return [email.strip() for email in to_emails_str.split(",")]
        return []

    async def send(self, message: NotificationMessage) -> bool:
        """Send email notification."""
        if not self.to_emails:
            logger.warning("No email recipients configured")
            return False

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            msg["Subject"] = f"[Agent Skeptic Bench] {message.title}"

            # Create email body
            body = self._create_email_body(message)
            msg.attach(MIMEText(body, "html"))

            # Send email
            await asyncio.to_thread(self._send_smtp, msg)

            logger.info(f"Email notification sent: {message.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    def _send_smtp(self, msg: MIMEMultipart) -> None:
        """Send email via SMTP (blocking operation)."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.use_tls:
                server.starttls()

            if self.smtp_user and self.smtp_password:
                server.login(self.smtp_user, self.smtp_password)

            server.send_message(msg)

    def _create_email_body(self, message: NotificationMessage) -> str:
        """Create HTML email body."""
        # Color scheme based on message level
        colors = {
            "info": "#2196F3",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336"
        }

        color = colors.get(message.level, colors["info"])

        body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{message.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 600px; margin: 0 auto; background-color: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .header {{ background-color: {color}; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; line-height: 1.6; }}
                .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
                .metadata {{ background-color: #f8f9fa; padding: 15px; margin: 15px 0; border-radius: 4px; }}
                .level-badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; color: white; font-size: 12px; text-transform: uppercase; background-color: {color}; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Agent Skeptic Bench</h1>
                    <span class="level-badge">{message.level}</span>
                </div>
                <div class="content">
                    <h2>{message.title}</h2>
                    <div>{message.content.replace(chr(10), '<br>')}</div>
        """

        if message.metadata:
            body += '<div class="metadata"><h3>Details</h3>'
            for key, value in message.metadata.items():
                body += f"<p><strong>{key.replace('_', ' ').title()}:</strong> {value}</p>"
            body += '</div>'

        body += """
                </div>
                <div class="footer">
                    <p>This is an automated notification from Agent Skeptic Bench</p>
                    <p>Terragon Labs - AI Safety Evaluation Framework</p>
                </div>
            </div>
        </body>
        </html>
        """

        return body

    async def is_available(self) -> bool:
        """Check if email notification is available."""
        return bool(self.smtp_host and self.to_emails)


class SlackNotifier(NotificationChannel):
    """Slack notification channel."""

    def __init__(self, webhook_url: str | None = None, channel: str | None = None):
        """Initialize Slack notifier.
        
        Args:
            webhook_url: Slack webhook URL
            channel: Slack channel (if not using webhook)
        """
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.channel = channel or os.getenv("SLACK_CHANNEL", "#agent-skeptic-bench")
        self.session = httpx.AsyncClient(timeout=30.0)

    async def send(self, message: NotificationMessage) -> bool:
        """Send Slack notification."""
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False

        try:
            # Create Slack message payload
            payload = self._create_slack_payload(message)

            # Send to Slack
            response = await self.session.post(self.webhook_url, json=payload)
            response.raise_for_status()

            logger.info(f"Slack notification sent: {message.title}")
            return True

        except httpx.HTTPError as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    def _create_slack_payload(self, message: NotificationMessage) -> dict[str, Any]:
        """Create Slack message payload."""
        # Color scheme based on message level
        colors = {
            "info": "#2196F3",
            "success": "good",
            "warning": "warning",
            "error": "danger"
        }

        color = colors.get(message.level, colors["info"])

        # Create attachment
        attachment = {
            "color": color,
            "title": message.title,
            "text": message.content,
            "footer": "Agent Skeptic Bench",
            "footer_icon": "https://agent-skeptic-bench.org/favicon.ico",
            "ts": int(asyncio.get_event_loop().time())
        }

        # Add metadata fields
        if message.metadata:
            fields = []
            for key, value in message.metadata.items():
                fields.append({
                    "title": key.replace("_", " ").title(),
                    "value": str(value),
                    "short": True
                })
            attachment["fields"] = fields

        return {
            "channel": self.channel,
            "username": "Agent Skeptic Bench",
            "icon_emoji": ":robot_face:",
            "attachments": [attachment]
        }

    async def is_available(self) -> bool:
        """Check if Slack notification is available."""
        return bool(self.webhook_url)

    async def close(self) -> None:
        """Close HTTP session."""
        await self.session.aclose()


class WebhookNotifier(NotificationChannel):
    """Generic webhook notification channel."""

    def __init__(self, webhook_url: str, headers: dict[str, str] | None = None):
        """Initialize webhook notifier.
        
        Args:
            webhook_url: Webhook URL
            headers: Optional HTTP headers
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.session = httpx.AsyncClient(timeout=30.0)

    async def send(self, message: NotificationMessage) -> bool:
        """Send webhook notification."""
        try:
            payload = {
                "title": message.title,
                "content": message.content,
                "level": message.level,
                "metadata": message.metadata,
                "timestamp": asyncio.get_event_loop().time(),
                "source": "agent-skeptic-bench"
            }

            response = await self.session.post(
                self.webhook_url,
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()

            logger.info(f"Webhook notification sent: {message.title}")
            return True

        except httpx.HTTPError as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False

    async def is_available(self) -> bool:
        """Check if webhook notification is available."""
        try:
            response = await self.session.head(self.webhook_url)
            return response.status_code < 500
        except:
            return False

    async def close(self) -> None:
        """Close HTTP session."""
        await self.session.aclose()


class NotificationManager:
    """Manages multiple notification channels."""

    def __init__(self):
        """Initialize notification manager."""
        self.channels: list[NotificationChannel] = []
        self._setup_default_channels()

    def _setup_default_channels(self) -> None:
        """Setup default notification channels based on environment."""
        # Email notifications
        if os.getenv("SMTP_HOST") and os.getenv("EMAIL_TO"):
            self.channels.append(EmailNotifier())

        # Slack notifications
        if os.getenv("SLACK_WEBHOOK_URL"):
            self.channels.append(SlackNotifier())

        # Custom webhook
        webhook_url = os.getenv("NOTIFICATION_WEBHOOK_URL")
        if webhook_url:
            headers = {}
            auth_header = os.getenv("NOTIFICATION_WEBHOOK_AUTH")
            if auth_header:
                headers["Authorization"] = auth_header
            self.channels.append(WebhookNotifier(webhook_url, headers))

    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self.channels.append(channel)

    async def send(self, message: NotificationMessage) -> int:
        """Send notification to all available channels.
        
        Returns:
            Number of channels that successfully sent the notification
        """
        if not self.channels:
            logger.warning("No notification channels configured")
            return 0

        tasks = []
        for channel in self.channels:
            if await channel.is_available():
                tasks.append(channel.send(message))

        if not tasks:
            logger.warning("No available notification channels")
            return 0

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for result in results if result is True)
        failed = len(results) - successful

        if failed > 0:
            logger.warning(f"Failed to send notification to {failed} channels")

        return successful

    async def send_evaluation_complete(self, session_id: str, results: dict[str, Any]) -> None:
        """Send evaluation completion notification."""
        message = NotificationMessage(
            title=f"Evaluation Complete: {session_id}",
            content=f"Benchmark evaluation has completed with a {results.get('pass_rate', 0):.1%} pass rate.",
            level="success" if results.get('pass_rate', 0) > 0.7 else "warning",
            metadata={
                "session_id": session_id,
                "total_scenarios": results.get('total_scenarios', 0),
                "pass_rate": f"{results.get('pass_rate', 0):.1%}",
                "overall_score": f"{results.get('overall_score', 0):.3f}",
                "agent_model": results.get('agent_model', 'Unknown')
            }
        )

        await self.send(message)

    async def send_evaluation_failed(self, session_id: str, error_message: str) -> None:
        """Send evaluation failure notification."""
        message = NotificationMessage(
            title=f"Evaluation Failed: {session_id}",
            content=f"Benchmark evaluation failed with error: {error_message}",
            level="error",
            metadata={
                "session_id": session_id,
                "error": error_message
            }
        )

        await self.send(message)

    async def send_system_alert(self, alert_type: str, description: str,
                               severity: str = "warning") -> None:
        """Send system alert notification."""
        message = NotificationMessage(
            title=f"System Alert: {alert_type}",
            content=description,
            level=severity,
            metadata={
                "alert_type": alert_type,
                "severity": severity
            }
        )

        await self.send(message)

    async def close(self) -> None:
        """Close all notification channels."""
        for channel in self.channels:
            if hasattr(channel, 'close'):
                await channel.close()


# Global notification manager instance
_notification_manager: NotificationManager | None = None


def get_notification_manager() -> NotificationManager:
    """Get global notification manager instance."""
    global _notification_manager

    if _notification_manager is None:
        _notification_manager = NotificationManager()

    return _notification_manager
