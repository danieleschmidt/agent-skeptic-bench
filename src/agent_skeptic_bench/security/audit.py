"""Security audit logging for Agent Skeptic Bench."""

import logging
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import threading
from collections import deque


logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events."""
    
    AUTHENTICATION_SUCCESS = "authentication_success"
    AUTHENTICATION_FAILURE = "authentication_failure"
    AUTHORIZATION_FAILURE = "authorization_failure"
    PASSWORD_CHANGE = "password_change"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    TOKEN_CREATED = "token_created"
    TOKEN_REVOKED = "token_revoked"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_REQUEST = "suspicious_request"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION_CHANGE = "configuration_change"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SECURITY_SCAN_DETECTED = "security_scan_detected"


class EventSeverity(Enum):
    """Security event severity levels."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event record."""
    
    id: str
    event_type: SecurityEventType
    severity: EventSeverity
    timestamp: datetime
    user_id: Optional[str]
    username: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: Optional[str]
    outcome: str  # "success", "failure", "blocked"
    details: Dict[str, Any]
    risk_score: float = 0.0
    session_id: Optional[str] = None
    request_id: Optional[str] = None


class AuditLogger:
    """Security audit logging system."""
    
    def __init__(self, max_events: int = 10000):
        """Initialize audit logger."""
        self.max_events = max_events
        self.events: deque = deque(maxlen=max_events)
        self.event_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        
        # Risk scoring weights
        self.risk_weights = {
            SecurityEventType.AUTHENTICATION_FAILURE: 2.0,
            SecurityEventType.AUTHORIZATION_FAILURE: 3.0,
            SecurityEventType.ACCOUNT_LOCKED: 5.0,
            SecurityEventType.RATE_LIMIT_EXCEEDED: 1.5,
            SecurityEventType.SUSPICIOUS_REQUEST: 4.0,
            SecurityEventType.PRIVILEGE_ESCALATION: 8.0,
            SecurityEventType.SECURITY_SCAN_DETECTED: 6.0
        }
    
    def log_event(self, event_type: SecurityEventType, severity: EventSeverity,
                  user_id: str = None, username: str = None, ip_address: str = None,
                  user_agent: str = None, resource: str = None, action: str = None,
                  outcome: str = "success", details: Dict[str, Any] = None,
                  session_id: str = None, request_id: str = None) -> SecurityEvent:
        """Log a security event."""
        
        import secrets
        
        event = SecurityEvent(
            id=secrets.token_urlsafe(16),
            event_type=event_type,
            severity=severity,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details or {},
            session_id=session_id,
            request_id=request_id
        )
        
        # Calculate risk score
        event.risk_score = self._calculate_risk_score(event)
        
        # Store event
        with self._lock:
            self.events.append(event)
            
            # Update event counts
            event_key = f"{event_type.value}:{outcome}"
            self.event_counts[event_key] = self.event_counts.get(event_key, 0) + 1
        
        # Log to standard logger based on severity
        log_level = {
            EventSeverity.LOW: logging.INFO,
            EventSeverity.MEDIUM: logging.WARNING,
            EventSeverity.HIGH: logging.ERROR,
            EventSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.INFO)
        
        log_message = (f"Security Event [{severity.value.upper()}] "
                      f"{event_type.value}: {outcome}")
        if username:
            log_message += f" - User: {username}"
        if ip_address:
            log_message += f" - IP: {ip_address}"
        
        logger.log(log_level, log_message)
        
        # Alert on high-risk events
        if event.risk_score > 7.0 or severity == EventSeverity.CRITICAL:
            self._trigger_security_alert(event)
        
        return event
    
    def _calculate_risk_score(self, event: SecurityEvent) -> float:
        """Calculate risk score for an event."""
        base_score = self.risk_weights.get(event.event_type, 1.0)
        
        # Severity multiplier
        severity_multiplier = {
            EventSeverity.LOW: 1.0,
            EventSeverity.MEDIUM: 2.0,
            EventSeverity.HIGH: 4.0,
            EventSeverity.CRITICAL: 8.0
        }.get(event.severity, 1.0)
        
        # Outcome modifier
        outcome_modifier = {
            "success": 1.0,
            "failure": 1.5,
            "blocked": 0.8
        }.get(event.outcome, 1.0)
        
        # Time-based modifier (recent events are riskier)
        time_modifier = 1.0  # Could implement time-based scoring
        
        # User pattern modifier
        user_modifier = self._get_user_risk_modifier(event.user_id, event.ip_address)
        
        risk_score = base_score * severity_multiplier * outcome_modifier * time_modifier * user_modifier
        
        return min(10.0, risk_score)  # Cap at 10.0
    
    def _get_user_risk_modifier(self, user_id: str, ip_address: str) -> float:
        """Get risk modifier based on user/IP patterns."""
        modifier = 1.0
        
        if user_id:
            # Check for repeated failures from this user
            recent_failures = self._count_recent_events(
                user_id=user_id,
                outcome="failure",
                hours=1
            )
            if recent_failures > 5:
                modifier *= 1.5
            elif recent_failures > 10:
                modifier *= 2.0
        
        if ip_address:
            # Check for suspicious IP activity
            recent_events = self._count_recent_events(
                ip_address=ip_address,
                hours=1
            )
            if recent_events > 100:
                modifier *= 1.3
            elif recent_events > 200:
                modifier *= 1.8
        
        return modifier
    
    def _count_recent_events(self, hours: int = 1, user_id: str = None,
                           ip_address: str = None, outcome: str = None) -> int:
        """Count recent events matching criteria."""
        cutoff = datetime.utcnow() - datetime.timedelta(hours=hours)
        count = 0
        
        with self._lock:
            for event in self.events:
                if event.timestamp < cutoff:
                    continue
                
                if user_id and event.user_id != user_id:
                    continue
                
                if ip_address and event.ip_address != ip_address:
                    continue
                
                if outcome and event.outcome != outcome:
                    continue
                
                count += 1
        
        return count
    
    def _trigger_security_alert(self, event: SecurityEvent) -> None:
        """Trigger security alert for high-risk events."""
        alert_message = (f"HIGH RISK SECURITY EVENT: {event.event_type.value} "
                        f"(Risk Score: {event.risk_score:.1f})")
        
        logger.critical(alert_message)
        
        # In a real implementation, this would integrate with alerting systems
        # like sending emails, Slack notifications, or triggering incident response
        
    def log_authentication_success(self, user_id: str, username: str,
                                 ip_address: str = None, user_agent: str = None,
                                 session_id: str = None) -> SecurityEvent:
        """Log successful authentication."""
        return self.log_event(
            event_type=SecurityEventType.AUTHENTICATION_SUCCESS,
            severity=EventSeverity.LOW,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            outcome="success",
            session_id=session_id,
            details={"method": "password"}
        )
    
    def log_authentication_failure(self, username: str, ip_address: str = None,
                                 user_agent: str = None, reason: str = None) -> SecurityEvent:
        """Log failed authentication."""
        return self.log_event(
            event_type=SecurityEventType.AUTHENTICATION_FAILURE,
            severity=EventSeverity.MEDIUM,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            outcome="failure",
            details={"reason": reason or "invalid_credentials"}
        )
    
    def log_authorization_failure(self, user_id: str, username: str,
                                resource: str, action: str, ip_address: str = None) -> SecurityEvent:
        """Log authorization failure."""
        return self.log_event(
            event_type=SecurityEventType.AUTHORIZATION_FAILURE,
            severity=EventSeverity.HIGH,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            resource=resource,
            action=action,
            outcome="blocked",
            details={"attempted_resource": resource, "attempted_action": action}
        )
    
    def log_rate_limit_exceeded(self, identifier: str, limit_type: str,
                              ip_address: str = None, user_id: str = None) -> SecurityEvent:
        """Log rate limit violation."""
        return self.log_event(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            severity=EventSeverity.MEDIUM,
            user_id=user_id,
            ip_address=ip_address,
            outcome="blocked",
            details={"identifier": identifier, "limit_type": limit_type}
        )
    
    def log_suspicious_request(self, description: str, ip_address: str = None,
                             user_agent: str = None, details: Dict[str, Any] = None) -> SecurityEvent:
        """Log suspicious request."""
        return self.log_event(
            event_type=SecurityEventType.SUSPICIOUS_REQUEST,
            severity=EventSeverity.HIGH,
            ip_address=ip_address,
            user_agent=user_agent,
            outcome="blocked",
            details={**(details or {}), "description": description}
        )
    
    def log_data_access(self, user_id: str, username: str, resource: str,
                       action: str = "read", ip_address: str = None) -> SecurityEvent:
        """Log data access."""
        severity = EventSeverity.LOW
        if "sensitive" in resource.lower() or "private" in resource.lower():
            severity = EventSeverity.MEDIUM
        
        return self.log_event(
            event_type=SecurityEventType.DATA_ACCESS,
            severity=severity,
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            resource=resource,
            action=action,
            outcome="success"
        )
    
    def log_configuration_change(self, user_id: str, username: str,
                                config_item: str, old_value: Any = None,
                                new_value: Any = None) -> SecurityEvent:
        """Log configuration change."""
        return self.log_event(
            event_type=SecurityEventType.CONFIGURATION_CHANGE,
            severity=EventSeverity.HIGH,
            user_id=user_id,
            username=username,
            resource=config_item,
            action="modify",
            outcome="success",
            details={
                "config_item": config_item,
                "old_value": str(old_value) if old_value is not None else None,
                "new_value": str(new_value) if new_value is not None else None
            }
        )
    
    def get_events(self, hours: int = 24, event_type: SecurityEventType = None,
                  severity: EventSeverity = None, user_id: str = None,
                  ip_address: str = None) -> List[SecurityEvent]:
        """Get security events matching criteria."""
        cutoff = datetime.utcnow() - datetime.timedelta(hours=hours)
        matching_events = []
        
        with self._lock:
            for event in self.events:
                if event.timestamp < cutoff:
                    continue
                
                if event_type and event.event_type != event_type:
                    continue
                
                if severity and event.severity != severity:
                    continue
                
                if user_id and event.user_id != user_id:
                    continue
                
                if ip_address and event.ip_address != ip_address:
                    continue
                
                matching_events.append(event)
        
        return sorted(matching_events, key=lambda x: x.timestamp, reverse=True)
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security event summary."""
        events = self.get_events(hours=hours)
        
        # Count by type
        type_counts = {}
        for event in events:
            type_counts[event.event_type.value] = type_counts.get(event.event_type.value, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for event in events:
            severity_counts[event.severity.value] = severity_counts.get(event.severity.value, 0) + 1
        
        # Count by outcome
        outcome_counts = {}
        for event in events:
            outcome_counts[event.outcome] = outcome_counts.get(event.outcome, 0) + 1
        
        # High-risk events
        high_risk_events = [e for e in events if e.risk_score > 5.0]
        
        # Top users/IPs by event count
        user_counts = {}
        ip_counts = {}
        for event in events:
            if event.username:
                user_counts[event.username] = user_counts.get(event.username, 0) + 1
            if event.ip_address:
                ip_counts[event.ip_address] = ip_counts.get(event.ip_address, 0) + 1
        
        return {
            "total_events": len(events),
            "time_period_hours": hours,
            "event_types": type_counts,
            "severity_distribution": severity_counts,
            "outcome_distribution": outcome_counts,
            "high_risk_events": len(high_risk_events),
            "average_risk_score": sum(e.risk_score for e in events) / len(events) if events else 0,
            "top_users": sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_ips": sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def export_events(self, hours: int = 24, format: str = "json") -> str:
        """Export security events."""
        events = self.get_events(hours=hours)
        
        if format == "json":
            return json.dumps([asdict(event) for event in events], 
                            default=str, indent=2)
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if events:
                fieldnames = asdict(events[0]).keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for event in events:
                    event_dict = asdict(event)
                    # Convert datetime and enum objects to strings
                    for key, value in event_dict.items():
                        if isinstance(value, (datetime, Enum)):
                            event_dict[key] = str(value)
                        elif isinstance(value, dict):
                            event_dict[key] = json.dumps(value)
                    writer.writerow(event_dict)
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger
    
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    
    return _audit_logger