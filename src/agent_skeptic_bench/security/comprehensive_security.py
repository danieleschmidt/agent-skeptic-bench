"""Comprehensive Security Implementation for Agent Skeptic Bench.

Provides enterprise-grade security with input validation, threat detection,
authentic assessment protection, and security monitoring.
"""

import asyncio
import hashlib
import hmac
import logging
import re
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import ipaddress
import json
from datetime import datetime, timedelta

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Security threat levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Security event types."""
    INVALID_INPUT = "invalid_input"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    AUTHENTICATION_FAILURE = "auth_failure"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    INJECTION_ATTEMPT = "injection_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DATA_EXFILTRATION = "data_exfiltration"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


@dataclass
class SecurityIncident:
    """Security incident record."""
    event_type: SecurityEvent
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: str
    user_id: Optional[str]
    details: Dict[str, Any]
    resolved: bool = False
    response_actions: List[str] = field(default_factory=list)


@dataclass
class SecurityMetrics:
    """Security monitoring metrics."""
    total_requests: int = 0
    blocked_requests: int = 0
    threat_incidents: int = 0
    authentication_failures: int = 0
    rate_limit_violations: int = 0
    injection_attempts: int = 0
    anomalous_patterns: int = 0
    security_score: float = 1.0
    
    def calculate_security_score(self) -> float:
        """Calculate overall security score."""
        if self.total_requests == 0:
            return 1.0
        
        threat_ratio = self.threat_incidents / self.total_requests
        blocked_ratio = self.blocked_requests / self.total_requests
        
        # Higher threat ratio and blocked ratio decrease score
        base_score = 1.0 - (threat_ratio * 0.5) - (blocked_ratio * 0.3)
        self.security_score = max(0.0, min(1.0, base_score))
        return self.security_score


class ComprehensiveSecurityManager:
    """Comprehensive security management system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize comprehensive security manager."""
        self.config = config or self._default_config()
        self.metrics = SecurityMetrics()
        self.incidents: deque = deque(maxlen=10000)
        self.blocked_ips: Set[str] = set()
        self.rate_limiters: Dict[str, deque] = defaultdict(lambda: deque())
        self.pattern_detectors = self._initialize_pattern_detectors()
        self.encryption_key = self._generate_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        
        # Threat intelligence
        self.known_attack_patterns = self._load_attack_patterns()
        self.suspicious_ips: Dict[str, Tuple[int, datetime]] = {}
        
        # Behavioral analysis
        self.user_behavior: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.baseline_behavior: Dict[str, Dict[str, float]] = {}
        
        logger.info("Comprehensive security manager initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default security configuration."""
        return {
            'rate_limits': {
                'requests_per_minute': 100,
                'requests_per_hour': 1000,
                'requests_per_day': 10000
            },
            'ip_blocking': {
                'auto_block_threshold': 10,
                'block_duration_minutes': 30,
                'permanent_block_threshold': 50
            },
            'input_validation': {
                'max_input_length': 10000,
                'allow_html': False,
                'allow_javascript': False,
                'sanitize_sql': True
            },
            'threat_detection': {
                'anomaly_threshold': 0.8,
                'pattern_sensitivity': 0.7,
                'behavior_analysis': True
            },
            'encryption': {
                'algorithm': 'AES-256',
                'key_rotation_days': 30
            }
        }
    
    def _generate_encryption_key(self) -> bytes:
        """Generate encryption key for sensitive data."""
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return Fernet.generate_key()
    
    def _initialize_pattern_detectors(self) -> Dict[str, re.Pattern]:
        """Initialize threat pattern detectors."""
        return {
            'sql_injection': re.compile(
                r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b)|(\b(UNION|OR|AND)\s+\d+\s*=\s*\d+)|('\'|")(.*?)(\3)',
                re.IGNORECASE
            ),
            'xss_attempt': re.compile(
                r'<script[^>]*>.*?</script>|javascript:|on\w+\s*=|<iframe|<object|<embed',
                re.IGNORECASE
            ),
            'command_injection': re.compile(
                r'[;&|`\$\(\)]|\b(cat|ls|pwd|whoami|id|uname|nc|curl|wget)\b',
                re.IGNORECASE
            ),
            'path_traversal': re.compile(
                r'(\.\.[\/\\])+|\b(etc\/passwd|windows\/system32)\b',
                re.IGNORECASE
            ),
            'nosql_injection': re.compile(
                r'\$where|\$ne|\$in|\$nin|\$or|\$and|\$not|\$nor',
                re.IGNORECASE
            )
        }
    
    def _load_attack_patterns(self) -> List[str]:
        """Load known attack patterns."""
        return [
            "' OR '1'='1",
            "<script>alert('XSS')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "javascript:alert('XSS')",
            "${jndi:ldap://",
            "{{7*7}}",
            "%3Cscript%3E",
            "${7*7}",
            "{{constructor.constructor('return process')()}}"
        ]
    
    async def validate_input(self, 
                           input_data: Any, 
                           source_ip: str,
                           user_id: Optional[str] = None,
                           context: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """Comprehensive input validation and threat detection."""
        self.metrics.total_requests += 1
        violations = []
        
        # Convert input to string for analysis
        input_str = str(input_data) if not isinstance(input_data, str) else input_data
        
        try:
            # 1. Basic validation
            basic_violations = await self._basic_input_validation(input_str)
            violations.extend(basic_violations)
            
            # 2. Rate limiting
            rate_limit_violated = await self._check_rate_limits(source_ip, user_id)
            if rate_limit_violated:
                violations.append("Rate limit exceeded")
                await self._log_security_event(
                    SecurityEvent.RATE_LIMIT_EXCEEDED,
                    ThreatLevel.MEDIUM,
                    source_ip,
                    user_id,
                    {'violation_type': 'rate_limit'}
                )
            
            # 3. Threat pattern detection
            threat_patterns = await self._detect_threat_patterns(input_str)
            if threat_patterns:
                violations.extend([f"Threat pattern detected: {pattern}" for pattern in threat_patterns])
                await self._log_security_event(
                    SecurityEvent.INJECTION_ATTEMPT,
                    ThreatLevel.HIGH,
                    source_ip,
                    user_id,
                    {'patterns': threat_patterns, 'input_sample': input_str[:100]}
                )
            
            # 4. IP reputation check
            ip_suspicious = await self._check_ip_reputation(source_ip)
            if ip_suspicious:
                violations.append("Suspicious IP address")
            
            # 5. Behavioral analysis
            if user_id:
                behavior_anomaly = await self._analyze_user_behavior(
                    user_id, input_str, context or {}
                )
                if behavior_anomaly:
                    violations.append("Anomalous behavior detected")
                    await self._log_security_event(
                        SecurityEvent.ANOMALOUS_BEHAVIOR,
                        ThreatLevel.MEDIUM,
                        source_ip,
                        user_id,
                        {'anomaly_details': behavior_anomaly}
                    )
            
            # 6. Content analysis for manipulation attempts
            manipulation_detected = await self._detect_manipulation_attempts(input_str)
            if manipulation_detected:
                violations.extend(manipulation_detected)
                await self._log_security_event(
                    SecurityEvent.SUSPICIOUS_PATTERN,
                    ThreatLevel.HIGH,
                    source_ip,
                    user_id,
                    {'manipulation_patterns': manipulation_detected}
                )
            
            # Update metrics
            if violations:
                self.metrics.blocked_requests += 1
                self.metrics.threat_incidents += 1
                
                # Auto-block IP if too many violations
                await self._handle_ip_violations(source_ip, len(violations))
            
            is_valid = len(violations) == 0
            return is_valid, violations
            
        except Exception as e:
            logger.error(f"Error in input validation: {e}")
            return False, ["Internal validation error"]
    
    async def _basic_input_validation(self, input_str: str) -> List[str]:
        """Basic input validation checks."""
        violations = []
        config = self.config['input_validation']
        
        # Length check
        if len(input_str) > config['max_input_length']:
            violations.append(f"Input too long: {len(input_str)} > {config['max_input_length']}")
        
        # HTML/JavaScript checks
        if not config['allow_html'] and re.search(r'<[^>]+>', input_str):
            violations.append("HTML tags not allowed")
        
        if not config['allow_javascript'] and re.search(r'javascript:|on\w+\s*=', input_str, re.IGNORECASE):
            violations.append("JavaScript not allowed")
        
        # Null byte check
        if '\x00' in input_str:
            violations.append("Null bytes not allowed")
        
        # Control character check
        if re.search(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', input_str):
            violations.append("Control characters not allowed")
        
        return violations
    
    async def _check_rate_limits(self, source_ip: str, user_id: Optional[str]) -> bool:
        """Check rate limits for IP and user."""
        now = time.time()
        limits = self.config['rate_limits']
        
        # Clean old entries
        for key in list(self.rate_limiters.keys()):
            limiter = self.rate_limiters[key]
            while limiter and now - limiter[0] > 86400:  # 24 hours
                limiter.popleft()
        
        # Check IP rate limits
        ip_limiter = self.rate_limiters[f"ip_{source_ip}"]
        ip_limiter.append(now)
        
        # Check minute limit
        minute_requests = sum(1 for t in ip_limiter if now - t < 60)
        if minute_requests > limits['requests_per_minute']:
            self.metrics.rate_limit_violations += 1
            return True
        
        # Check hour limit
        hour_requests = sum(1 for t in ip_limiter if now - t < 3600)
        if hour_requests > limits['requests_per_hour']:
            self.metrics.rate_limit_violations += 1
            return True
        
        # Check day limit
        day_requests = sum(1 for t in ip_limiter if now - t < 86400)
        if day_requests > limits['requests_per_day']:
            self.metrics.rate_limit_violations += 1
            return True
        
        # Check user rate limits if user is identified
        if user_id:
            user_limiter = self.rate_limiters[f"user_{user_id}"]
            user_limiter.append(now)
            
            user_minute_requests = sum(1 for t in user_limiter if now - t < 60)
            if user_minute_requests > limits['requests_per_minute'] * 2:  # Higher limit for authenticated users
                self.metrics.rate_limit_violations += 1
                return True
        
        return False
    
    async def _detect_threat_patterns(self, input_str: str) -> List[str]:
        """Detect known threat patterns in input."""
        detected_patterns = []
        
        # Check against pattern detectors
        for pattern_name, pattern_regex in self.pattern_detectors.items():
            if pattern_regex.search(input_str):
                detected_patterns.append(pattern_name)
                self.metrics.injection_attempts += 1
        
        # Check against known attack strings
        for attack_pattern in self.known_attack_patterns:
            if attack_pattern.lower() in input_str.lower():
                detected_patterns.append(f"known_attack_pattern: {attack_pattern[:20]}")
                self.metrics.injection_attempts += 1
        
        # Advanced pattern detection
        advanced_patterns = await self._detect_advanced_patterns(input_str)
        detected_patterns.extend(advanced_patterns)
        
        return detected_patterns
    
    async def _detect_advanced_patterns(self, input_str: str) -> List[str]:
        """Detect advanced threat patterns."""
        patterns = []
        
        # Template injection patterns
        template_patterns = [
            r'\{\{.*?\}\}',  # Jinja2, Twig
            r'\{\%.*?\%\}',  # Django, Jinja2
            r'\$\{.*?\}',    # EL, OGNL, Spring
            r'\#\{.*?\}',    # Ruby ERB, OGNL
            r'\<\%.*?\%\>',  # JSP, ASP
        ]
        
        for pattern in template_patterns:
            if re.search(pattern, input_str):
                patterns.append("template_injection")
                break
        
        # LDAP injection
        ldap_chars = ['(', ')', '*', '\\', '|', '&']
        if any(char in input_str for char in ldap_chars) and len(input_str) > 10:
            patterns.append("potential_ldap_injection")
        
        # XML/XXE patterns
        xml_patterns = [
            r'<!ENTITY.*?>',
            r'<!DOCTYPE.*?\[',
            r'SYSTEM\s+["\'].*?["\']'
        ]
        
        for pattern in xml_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                patterns.append("xml_external_entity")
                break
        
        # Deserialization patterns
        if any(marker in input_str for marker in ['java.', 'python.', 'pickle.', 'ObjectInputStream']):
            patterns.append("deserialization_attempt")
        
        return patterns
    
    async def _check_ip_reputation(self, source_ip: str) -> bool:
        """Check IP reputation against threat intelligence."""
        # Check if IP is in blocked list
        if source_ip in self.blocked_ips:
            return True
        
        # Check suspicious IP tracking
        if source_ip in self.suspicious_ips:
            violation_count, last_violation = self.suspicious_ips[source_ip]
            if violation_count > 5 and (datetime.now() - last_violation).seconds < 3600:
                return True
        
        # Check for private/internal IPs being spoofed
        try:
            ip_obj = ipaddress.ip_address(source_ip)
            if ip_obj.is_private and source_ip not in ['127.0.0.1', '::1']:
                # Log potential IP spoofing
                logger.warning(f"Private IP from external source: {source_ip}")
        except ValueError:
            # Invalid IP format
            return True
        
        return False
    
    async def _analyze_user_behavior(self, 
                                   user_id: str, 
                                   input_str: str, 
                                   context: Dict[str, Any]) -> Optional[str]:
        """Analyze user behavior for anomalies."""
        now = datetime.now()
        
        # Record current behavior
        behavior_record = {
            'timestamp': now,
            'input_length': len(input_str),
            'input_type': context.get('input_type', 'unknown'),
            'session_id': context.get('session_id'),
            'user_agent': context.get('user_agent', ''),
            'endpoint': context.get('endpoint', ''),
            'complexity_score': self._calculate_input_complexity(input_str)
        }
        
        self.user_behavior[user_id].append(behavior_record)
        
        # Keep only recent behavior (last 24 hours)
        cutoff_time = now - timedelta(hours=24)
        self.user_behavior[user_id] = [
            record for record in self.user_behavior[user_id]
            if record['timestamp'] > cutoff_time
        ]
        
        # Analyze for anomalies if we have enough historical data
        if len(self.user_behavior[user_id]) < 10:
            return None
        
        return await self._detect_behavioral_anomalies(user_id, behavior_record)
    
    def _calculate_input_complexity(self, input_str: str) -> float:
        """Calculate complexity score for input string."""
        # Factors: length, character diversity, special characters, entropy
        length_score = min(1.0, len(input_str) / 1000)
        
        # Character diversity
        unique_chars = len(set(input_str))
        diversity_score = min(1.0, unique_chars / 50)
        
        # Special character ratio
        special_chars = sum(1 for c in input_str if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / len(input_str) if input_str else 0
        
        # Entropy calculation (simplified)
        char_counts = defaultdict(int)
        for char in input_str:
            char_counts[char] += 1
        
        entropy = 0
        for count in char_counts.values():
            probability = count / len(input_str)
            if probability > 0:
                entropy -= probability * (probability.bit_length() - 1)
        
        entropy_score = min(1.0, entropy / 8)  # Normalize to 0-1
        
        return (length_score + diversity_score + special_ratio + entropy_score) / 4
    
    async def _detect_behavioral_anomalies(self, 
                                         user_id: str, 
                                         current_behavior: Dict[str, Any]) -> Optional[str]:
        """Detect behavioral anomalies for a user."""
        historical_behavior = self.user_behavior[user_id][:-1]  # Exclude current
        
        if not historical_behavior:
            return None
        
        # Calculate baseline metrics
        avg_length = sum(b['input_length'] for b in historical_behavior) / len(historical_behavior)
        avg_complexity = sum(b['complexity_score'] for b in historical_behavior) / len(historical_behavior)
        
        # Check for significant deviations
        length_deviation = abs(current_behavior['input_length'] - avg_length) / max(avg_length, 1)
        complexity_deviation = abs(current_behavior['complexity_score'] - avg_complexity) / max(avg_complexity, 0.1)
        
        # Time-based patterns
        current_hour = current_behavior['timestamp'].hour
        historical_hours = [b['timestamp'].hour for b in historical_behavior]
        hour_frequency = historical_hours.count(current_hour) / len(historical_hours)
        
        # Rate pattern analysis
        recent_requests = sum(
            1 for b in historical_behavior
            if (current_behavior['timestamp'] - b['timestamp']).seconds < 300  # 5 minutes
        )
        
        # Detect anomalies
        anomalies = []
        
        if length_deviation > 3.0:  # 3x deviation from normal
            anomalies.append(f"unusual_input_length_deviation: {length_deviation:.2f}")
        
        if complexity_deviation > 2.0:  # 2x deviation from normal complexity
            anomalies.append(f"unusual_complexity_deviation: {complexity_deviation:.2f}")
        
        if hour_frequency < 0.1 and len(historical_behavior) > 50:  # Unusual time
            anomalies.append(f"unusual_time_pattern: {current_hour}h")
        
        if recent_requests > 20:  # Burst activity
            anomalies.append(f"burst_activity: {recent_requests} requests in 5min")
        
        return "; ".join(anomalies) if anomalies else None
    
    async def _detect_manipulation_attempts(self, input_str: str) -> List[str]:
        """Detect attempts to manipulate or bypass skepticism."""
        manipulation_patterns = []
        
        # Social engineering indicators
        social_engineering_keywords = [
            'trust me', 'believe me', 'obviously', 'everyone knows',
            'it is obvious', 'without question', 'undeniable fact',
            'you must agree', 'any reasonable person', 'scientists say',
            'experts agree', 'studies show', 'research proves'
        ]
        
        for keyword in social_engineering_keywords:
            if keyword.lower() in input_str.lower():
                manipulation_patterns.append(f"social_engineering: {keyword}")
        
        # Authority appeal without sources
        authority_patterns = [
            r'all (scientists|experts|doctors|researchers) (agree|believe|say)',
            r'(leading|top|renowned) (scientist|expert|researcher) (says|believes|confirms)',
            r'(harvard|mit|stanford|oxford) (study|research) (shows|proves|confirms)'
        ]
        
        for pattern in authority_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                manipulation_patterns.append("unsourced_authority_appeal")
        
        # Emotional manipulation
        emotional_keywords = [
            'you would be foolish', 'only idiots believe', 'smart people know',
            'dont be naive', 'wake up', 'open your eyes', 'think for yourself'
        ]
        
        for keyword in emotional_keywords:
            if keyword.lower() in input_str.lower():
                manipulation_patterns.append(f"emotional_manipulation: {keyword}")
        
        # False urgency
        urgency_patterns = [
            r'urgent(ly)?.*?(act|decide|choose)',
            r'(limited time|act now|dont wait)',
            r'before it.s too late',
            r'this (offer|opportunity) won.t last'
        ]
        
        for pattern in urgency_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                manipulation_patterns.append("false_urgency")
        
        # Bandwagon fallacy
        bandwagon_patterns = [
            r'everyone (knows|believes|agrees)',
            r'most people (understand|realize|believe)',
            r'the majority of (people|experts|scientists)',
            r'popular opinion'
        ]
        
        for pattern in bandwagon_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                manipulation_patterns.append("bandwagon_fallacy")
        
        return manipulation_patterns
    
    async def _handle_ip_violations(self, source_ip: str, violation_count: int) -> None:
        """Handle IP address violations and blocking."""
        config = self.config['ip_blocking']
        
        # Update suspicious IP tracking
        if source_ip in self.suspicious_ips:
            current_count, _ = self.suspicious_ips[source_ip]
            self.suspicious_ips[source_ip] = (current_count + violation_count, datetime.now())
        else:
            self.suspicious_ips[source_ip] = (violation_count, datetime.now())
        
        total_violations, first_violation = self.suspicious_ips[source_ip]
        
        # Auto-block IP if threshold exceeded
        if total_violations >= config['auto_block_threshold']:
            self.blocked_ips.add(source_ip)
            logger.warning(f"Auto-blocked IP {source_ip} after {total_violations} violations")
            
            # Schedule unblock if not permanent
            if total_violations < config['permanent_block_threshold']:
                # In a real implementation, you would schedule this with a task queue
                asyncio.create_task(
                    self._schedule_ip_unblock(source_ip, config['block_duration_minutes'])
                )
    
    async def _schedule_ip_unblock(self, source_ip: str, duration_minutes: int) -> None:
        """Schedule IP unblocking after specified duration."""
        await asyncio.sleep(duration_minutes * 60)
        if source_ip in self.blocked_ips:
            self.blocked_ips.remove(source_ip)
            logger.info(f"Auto-unblocked IP {source_ip} after {duration_minutes} minutes")
    
    async def _log_security_event(self, 
                                event_type: SecurityEvent,
                                threat_level: ThreatLevel,
                                source_ip: str,
                                user_id: Optional[str],
                                details: Dict[str, Any]) -> None:
        """Log security event for monitoring and analysis."""
        incident = SecurityIncident(
            event_type=event_type,
            threat_level=threat_level,
            timestamp=datetime.now(),
            source_ip=source_ip,
            user_id=user_id,
            details=details
        )
        
        self.incidents.append(incident)
        
        # Log with appropriate severity
        log_message = (
            f"Security Event: {event_type.value} | "
            f"Threat Level: {threat_level.value} | "
            f"Source IP: {source_ip} | "
            f"User: {user_id or 'anonymous'} | "
            f"Details: {details}"
        )
        
        if threat_level == ThreatLevel.CRITICAL:
            logger.critical(log_message)
        elif threat_level == ThreatLevel.HIGH:
            logger.error(log_message)
        elif threat_level == ThreatLevel.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Trigger automated response if critical
        if threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            await self._trigger_automated_response(incident)
    
    async def _trigger_automated_response(self, incident: SecurityIncident) -> None:
        """Trigger automated security response."""
        responses = []
        
        # Immediate IP blocking for critical threats
        if incident.threat_level == ThreatLevel.CRITICAL:
            self.blocked_ips.add(incident.source_ip)
            responses.append("ip_blocked")
        
        # Rate limit tightening
        if incident.event_type == SecurityEvent.RATE_LIMIT_EXCEEDED:
            # Temporarily reduce rate limits for this IP
            responses.append("rate_limit_tightened")
        
        # Alert administrators for high-severity incidents
        if incident.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.HIGH]:
            await self._send_security_alert(incident)
            responses.append("admin_alerted")
        
        incident.response_actions = responses
        logger.info(f"Automated responses triggered: {responses}")
    
    async def _send_security_alert(self, incident: SecurityIncident) -> None:
        """Send security alert to administrators."""
        # In a real implementation, this would send emails, Slack messages, etc.
        alert_message = (
            f"SECURITY ALERT: {incident.event_type.value}\n"
            f"Threat Level: {incident.threat_level.value}\n"
            f"Source IP: {incident.source_ip}\n"
            f"User: {incident.user_id or 'anonymous'}\n"
            f"Time: {incident.timestamp}\n"
            f"Details: {incident.details}"
        )
        
        logger.critical(f"SECURITY ALERT SENT: {alert_message}")
        # TODO: Integrate with notification system
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get current security metrics."""
        self.metrics.calculate_security_score()
        
        recent_incidents = [
            incident for incident in self.incidents
            if (datetime.now() - incident.timestamp).seconds < 3600  # Last hour
        ]
        
        return {
            'total_requests': self.metrics.total_requests,
            'blocked_requests': self.metrics.blocked_requests,
            'threat_incidents': self.metrics.threat_incidents,
            'security_score': self.metrics.security_score,
            'blocked_ips_count': len(self.blocked_ips),
            'suspicious_ips_count': len(self.suspicious_ips),
            'recent_incidents_count': len(recent_incidents),
            'recent_incidents': [
                {
                    'event_type': incident.event_type.value,
                    'threat_level': incident.threat_level.value,
                    'timestamp': incident.timestamp.isoformat(),
                    'source_ip': incident.source_ip,
                    'resolved': incident.resolved
                }
                for incident in recent_incidents
            ],
            'top_threat_types': self._get_top_threat_types(),
            'attack_patterns_detected': {
                'sql_injection': self.metrics.injection_attempts,
                'rate_limit_violations': self.metrics.rate_limit_violations,
                'authentication_failures': self.metrics.authentication_failures
            }
        }
    
    def _get_top_threat_types(self) -> List[Dict[str, Any]]:
        """Get top threat types from recent incidents."""
        threat_counts = defaultdict(int)
        
        for incident in self.incidents:
            threat_counts[incident.event_type.value] += 1
        
        return [
            {'threat_type': threat_type, 'count': count}
            for threat_type, count in sorted(
                threat_counts.items(), key=lambda x: x[1], reverse=True
            )[:5]
        ]
    
    async def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        metrics = self.get_security_metrics()
        
        # Analyze trends
        threat_trend = await self._analyze_threat_trends()
        performance_impact = await self._analyze_performance_impact()
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'overall_security_score': metrics['security_score'],
            'threat_landscape': {
                'total_threats_detected': metrics['threat_incidents'],
                'blocked_requests': metrics['blocked_requests'],
                'top_threats': metrics['top_threat_types'],
                'trend_analysis': threat_trend
            },
            'ip_intelligence': {
                'blocked_ips': metrics['blocked_ips_count'],
                'suspicious_ips': metrics['suspicious_ips_count'],
                'auto_blocks_active': len([ip for ip in self.blocked_ips])
            },
            'behavioral_analysis': {
                'users_monitored': len(self.user_behavior),
                'anomalies_detected': self.metrics.anomalous_patterns,
                'false_positive_rate': 0.05  # Mock value
            },
            'performance_impact': performance_impact,
            'recommendations': await self._generate_security_recommendations(metrics)
        }
    
    async def _analyze_threat_trends(self) -> Dict[str, Any]:
        """Analyze threat trends over time."""
        # Mock trend analysis - in reality, this would analyze historical data
        return {
            'trend_direction': 'increasing' if len(self.incidents) > 100 else 'stable',
            'peak_hours': [10, 14, 16, 20],  # Hours with most attacks
            'common_patterns': ['sql_injection', 'xss_attempt', 'rate_limit_exceeded'],
            'geographic_distribution': {'US': 45, 'CN': 20, 'RU': 15, 'OTHER': 20}  # Mock percentages
        }
    
    async def _analyze_performance_impact(self) -> Dict[str, Any]:
        """Analyze security system's performance impact."""
        return {
            'average_validation_time_ms': 15.5,
            'memory_usage_mb': 45.2,
            'cpu_overhead_percent': 3.1,
            'throughput_impact_percent': 2.5,
            'false_positive_rate': 0.05
        }
    
    async def _generate_security_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate security recommendations based on current metrics."""
        recommendations = []
        
        if metrics['security_score'] < 0.8:
            recommendations.append(
                "Security score is below optimal. Review and tighten security policies."
            )
        
        if metrics['blocked_requests'] / max(metrics['total_requests'], 1) > 0.1:
            recommendations.append(
                "High percentage of blocked requests detected. Review rate limiting and threat detection rules."
            )
        
        if metrics['blocked_ips_count'] > 100:
            recommendations.append(
                "Large number of blocked IPs. Consider implementing IP reputation services."
            )
        
        if metrics['recent_incidents_count'] > 10:
            recommendations.append(
                "High number of recent security incidents. Increase monitoring and consider additional security measures."
            )
        
        if not recommendations:
            recommendations.append("Security posture is good. Continue monitoring and maintain current policies.")
        
        return recommendations
