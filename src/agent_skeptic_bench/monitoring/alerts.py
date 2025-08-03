"""Alert management system for Agent Skeptic Bench."""

import logging
import asyncio
import smtplib
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiohttp
import json


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(Enum):
    """Notification channels."""
    
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    CONSOLE = "console"


@dataclass
class AlertCondition:
    """Alert condition definition."""
    
    metric_name: str
    operator: str  # ">", "<", ">=", "<=", "==", "!="
    threshold: float
    duration: int  # seconds
    description: str


@dataclass
class AlertRule:
    """Alert rule definition."""
    
    id: str
    name: str
    description: str
    conditions: List[AlertCondition]
    severity: AlertSeverity
    notification_channels: List[NotificationChannel]
    enabled: bool = True
    cooldown_period: int = 300  # seconds
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert instance."""
    
    id: str
    rule_id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metric_values: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    notification_sent: bool = False


@dataclass
class NotificationConfig:
    """Notification configuration."""
    
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)
    
    slack_webhook_url: str = ""
    slack_channel: str = "#alerts"
    
    webhook_urls: List[str] = field(default_factory=list)
    
    sms_api_key: str = ""
    sms_numbers: List[str] = field(default_factory=list)


class AlertManager:
    """Comprehensive alert management system."""
    
    def __init__(self, notification_config: NotificationConfig = None):
        """Initialize alert manager."""
        self.notification_config = notification_config or NotificationConfig()
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_rule_evaluation: Dict[str, datetime] = {}
        self.condition_states: Dict[str, Dict[str, Any]] = {}  # Track condition duration
        
        # Notification handlers
        self.notification_handlers: Dict[NotificationChannel, Callable] = {
            NotificationChannel.EMAIL: self._send_email_notification,
            NotificationChannel.SLACK: self._send_slack_notification,
            NotificationChannel.WEBHOOK: self._send_webhook_notification,
            NotificationChannel.CONSOLE: self._send_console_notification,
        }
        
        # Monitoring task
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.alert_rules[rule.id] = rule
        self.condition_states[rule.id] = {}
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_id: str) -> None:
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            if rule_id in self.condition_states:
                del self.condition_states[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def enable_rule(self, rule_id: str) -> None:
        """Enable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = True
            logger.info(f"Enabled alert rule: {rule_id}")
    
    def disable_rule(self, rule_id: str) -> None:
        """Disable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = False
            logger.info(f"Disabled alert rule: {rule_id}")
    
    async def evaluate_rules(self, metrics: Dict[str, float]) -> List[Alert]:
        """Evaluate alert rules against current metrics."""
        triggered_alerts = []
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown period
            last_eval = self.last_rule_evaluation.get(rule_id)
            if last_eval and (datetime.utcnow() - last_eval).total_seconds() < rule.cooldown_period:
                continue
            
            # Evaluate conditions
            alert = await self._evaluate_rule_conditions(rule, metrics)
            if alert:
                triggered_alerts.append(alert)
                self.last_rule_evaluation[rule_id] = datetime.utcnow()
        
        return triggered_alerts
    
    async def _evaluate_rule_conditions(self, rule: AlertRule, metrics: Dict[str, float]) -> Optional[Alert]:
        """Evaluate conditions for a specific rule."""
        rule_state = self.condition_states.get(rule.id, {})
        all_conditions_met = True
        current_time = datetime.utcnow()
        
        for condition in rule.conditions:
            metric_value = metrics.get(condition.metric_name)
            
            if metric_value is None:
                all_conditions_met = False
                continue
            
            # Check if condition is met
            condition_met = self._evaluate_condition(condition, metric_value)
            
            if condition_met:
                # Start tracking duration if not already tracking
                if condition.metric_name not in rule_state:
                    rule_state[condition.metric_name] = {
                        'start_time': current_time,
                        'value': metric_value
                    }
                
                # Check if duration threshold is met
                duration = (current_time - rule_state[condition.metric_name]['start_time']).total_seconds()
                if duration < condition.duration:
                    all_conditions_met = False
            else:
                # Reset condition state
                if condition.metric_name in rule_state:
                    del rule_state[condition.metric_name]
                all_conditions_met = False
        
        self.condition_states[rule.id] = rule_state
        
        # Create alert if all conditions are met
        if all_conditions_met:
            # Check if alert already exists
            existing_alert_id = f"{rule.id}_{int(current_time.timestamp())}"
            if existing_alert_id not in self.active_alerts:
                alert = Alert(
                    id=existing_alert_id,
                    rule_id=rule.id,
                    name=rule.name,
                    description=rule.description,
                    severity=rule.severity,
                    status=AlertStatus.ACTIVE,
                    triggered_at=current_time,
                    metric_values=metrics.copy(),
                    tags=rule.tags.copy()
                )
                
                self.active_alerts[alert.id] = alert
                self.alert_history.append(alert)
                
                # Send notifications
                await self._send_notifications(alert, rule)
                
                return alert
        
        return None
    
    def _evaluate_condition(self, condition: AlertCondition, value: float) -> bool:
        """Evaluate a single condition."""
        if condition.operator == ">":
            return value > condition.threshold
        elif condition.operator == "<":
            return value < condition.threshold
        elif condition.operator == ">=":
            return value >= condition.threshold
        elif condition.operator == "<=":
            return value <= condition.threshold
        elif condition.operator == "==":
            return abs(value - condition.threshold) < 1e-9
        elif condition.operator == "!=":
            return abs(value - condition.threshold) >= 1e-9
        else:
            logger.warning(f"Unknown operator: {condition.operator}")
            return False
    
    async def _send_notifications(self, alert: Alert, rule: AlertRule) -> None:
        """Send notifications for an alert."""
        for channel in rule.notification_channels:
            try:
                handler = self.notification_handlers.get(channel)
                if handler:
                    await handler(alert)
                    alert.notification_sent = True
                else:
                    logger.warning(f"No handler for notification channel: {channel}")
            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification: {e}")
    
    async def _send_email_notification(self, alert: Alert) -> None:
        """Send email notification."""
        if not self.notification_config.email_to:
            return
        
        subject = f"[{alert.severity.value.upper()}] {alert.name}"
        
        body = f"""
Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Triggered: {alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

Description: {alert.description}

Metric Values:
"""
        for metric, value in alert.metric_values.items():
            body += f"  {metric}: {value}\n"
        
        if alert.tags:
            body += "\nTags:\n"
            for key, value in alert.tags.items():
                body += f"  {key}: {value}\n"
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.notification_config.email_from
            msg['To'] = ", ".join(self.notification_config.email_to)
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.notification_config.email_smtp_server, 
                                self.notification_config.email_smtp_port)
            server.starttls()
            server.login(self.notification_config.email_username, 
                        self.notification_config.email_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent for alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    async def _send_slack_notification(self, alert: Alert) -> None:
        """Send Slack notification."""
        if not self.notification_config.slack_webhook_url:
            return
        
        color = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "danger",
            AlertSeverity.CRITICAL: "danger"
        }.get(alert.severity, "danger")
        
        payload = {
            "channel": self.notification_config.slack_channel,
            "username": "Agent Skeptic Bench",
            "icon_emoji": ":warning:",
            "attachments": [
                {
                    "color": color,
                    "title": f"Alert: {alert.name}",
                    "text": alert.description,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                        {"title": "Triggered", "value": alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": True}
                    ]
                }
            ]
        }
        
        # Add metric values
        if alert.metric_values:
            metrics_text = "\n".join([f"{k}: {v}" for k, v in alert.metric_values.items()])
            payload["attachments"][0]["fields"].append({
                "title": "Metrics",
                "value": f"```{metrics_text}```",
                "short": False
            })
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.notification_config.slack_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert: {alert.id}")
                    else:
                        logger.error(f"Slack notification failed with status: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    async def _send_webhook_notification(self, alert: Alert) -> None:
        """Send webhook notification."""
        if not self.notification_config.webhook_urls:
            return
        
        payload = {
            "alert_id": alert.id,
            "rule_id": alert.rule_id,
            "name": alert.name,
            "description": alert.description,
            "severity": alert.severity.value,
            "status": alert.status.value,
            "triggered_at": alert.triggered_at.isoformat(),
            "metric_values": alert.metric_values,
            "tags": alert.tags
        }
        
        for webhook_url in self.notification_config.webhook_urls:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Webhook notification sent to {webhook_url} for alert: {alert.id}")
                        else:
                            logger.error(f"Webhook notification to {webhook_url} failed with status: {response.status}")
                            
            except Exception as e:
                logger.error(f"Failed to send webhook notification to {webhook_url}: {e}")
    
    async def _send_console_notification(self, alert: Alert) -> None:
        """Send console notification (log)."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        message = f"ALERT [{alert.severity.value.upper()}] {alert.name}: {alert.description}"
        logger.log(log_level, message)
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.utcnow()
            alert.acknowledged_by = acknowledged_by
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = datetime.utcnow()
            del self.active_alerts[alert_id]
            logger.info(f"Alert resolved: {alert_id}")
            return True
        return False
    
    def suppress_alert(self, alert_id: str) -> bool:
        """Suppress an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            logger.info(f"Alert suppressed: {alert_id}")
            return True
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.triggered_at, reverse=True)
    
    def get_alert_history(self, hours: int = 24, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        alerts = [
            alert for alert in self.alert_history 
            if alert.triggered_at > cutoff
        ]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return sorted(alerts, key=lambda x: x.triggered_at, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_alerts = list(self.active_alerts.values())
        recent_alerts = self.get_alert_history(hours=24)
        
        # Count by severity
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        # Count by status
        status_counts = {status.value: 0 for status in AlertStatus}
        for alert in recent_alerts:
            status_counts[alert.status.value] += 1
        
        return {
            "active_alerts": len(active_alerts),
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "alerts_24h": len(recent_alerts),
            "severity_distribution": severity_counts,
            "status_distribution": status_counts,
            "top_alerting_rules": self._get_top_alerting_rules()
        }
    
    def _get_top_alerting_rules(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get rules that trigger alerts most frequently."""
        rule_counts = {}
        
        for alert in self.alert_history:
            rule_id = alert.rule_id
            if rule_id not in rule_counts:
                rule_counts[rule_id] = {"count": 0, "rule_name": alert.name}
            rule_counts[rule_id]["count"] += 1
        
        sorted_rules = sorted(rule_counts.items(), key=lambda x: x[1]["count"], reverse=True)
        
        return [
            {"rule_id": rule_id, "rule_name": data["rule_name"], "alert_count": data["count"]}
            for rule_id, data in sorted_rules[:limit]
        ]


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    
    if _alert_manager is None:
        _alert_manager = AlertManager()
    
    return _alert_manager


def setup_default_alert_rules() -> None:
    """Setup default alert rules."""
    manager = get_alert_manager()
    
    # High CPU usage alert
    manager.add_alert_rule(AlertRule(
        id="high_cpu_usage",
        name="High CPU Usage",
        description="CPU usage is consistently high",
        conditions=[
            AlertCondition(
                metric_name="cpu_usage",
                operator=">",
                threshold=80.0,
                duration=300,  # 5 minutes
                description="CPU usage > 80% for 5 minutes"
            )
        ],
        severity=AlertSeverity.WARNING,
        notification_channels=[NotificationChannel.CONSOLE, NotificationChannel.EMAIL]
    ))
    
    # Critical CPU usage alert
    manager.add_alert_rule(AlertRule(
        id="critical_cpu_usage",
        name="Critical CPU Usage",
        description="CPU usage is critically high",
        conditions=[
            AlertCondition(
                metric_name="cpu_usage",
                operator=">",
                threshold=95.0,
                duration=60,  # 1 minute
                description="CPU usage > 95% for 1 minute"
            )
        ],
        severity=AlertSeverity.CRITICAL,
        notification_channels=[NotificationChannel.CONSOLE, NotificationChannel.EMAIL, NotificationChannel.SLACK]
    ))
    
    # High memory usage alert
    manager.add_alert_rule(AlertRule(
        id="high_memory_usage",
        name="High Memory Usage",
        description="Memory usage is consistently high",
        conditions=[
            AlertCondition(
                metric_name="memory_usage",
                operator=">",
                threshold=85.0,
                duration=300,  # 5 minutes
                description="Memory usage > 85% for 5 minutes"
            )
        ],
        severity=AlertSeverity.WARNING,
        notification_channels=[NotificationChannel.CONSOLE, NotificationChannel.EMAIL]
    ))
    
    # High error rate alert
    manager.add_alert_rule(AlertRule(
        id="high_error_rate",
        name="High Error Rate",
        description="Error rate is too high",
        conditions=[
            AlertCondition(
                metric_name="error_rate",
                operator=">",
                threshold=0.1,  # 10%
                duration=120,  # 2 minutes
                description="Error rate > 10% for 2 minutes"
            )
        ],
        severity=AlertSeverity.ERROR,
        notification_channels=[NotificationChannel.CONSOLE, NotificationChannel.EMAIL, NotificationChannel.SLACK]
    ))
    
    logger.info("Default alert rules configured")