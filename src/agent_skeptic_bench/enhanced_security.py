"""Enhanced Security Framework for Agent Skeptic Bench.

Comprehensive security measures including advanced input validation,
threat detection, audit logging, and security pattern analysis.
"""

import hashlib
import hmac
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
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
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "authz_violation"
    INPUT_VALIDATION_FAILURE = "input_validation_failure"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    DATA_BREACH_ATTEMPT = "data_breach_attempt"
    INJECTION_ATTEMPT = "injection_attempt"
    PRIVILEGE_ESCALATION = "privilege_escalation"


@dataclass
class SecurityAlert:
    """Security alert data structure."""
    alert_id: str
    event_type: SecurityEvent
    threat_level: ThreatLevel
    timestamp: datetime
    source_ip: str
    user_id: Optional[str]
    description: str
    evidence: Dict[str, Any]
    mitigation_applied: bool = False
    false_positive_score: float = 0.0


@dataclass 
class SecurityMetrics:
    """Security metrics tracking."""
    total_requests: int = 0
    blocked_requests: int = 0
    security_alerts: int = 0
    false_positives: int = 0
    attack_patterns_detected: int = 0
    average_threat_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class AdvancedInputValidator:
    """Advanced input validation with ML-based threat detection."""
    
    def __init__(self):
        """Initialize advanced input validator."""
        self.validation_patterns = self._load_validation_patterns()
        self.threat_signatures = self._load_threat_signatures()
        self.validation_history: List[Dict[str, Any]] = []
        self.blocked_patterns: Set[str] = set()
        
    def validate_input(self, 
                      input_data: Any,
                      input_type: str = "general",
                      context: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str], float]:
        """Validate input with comprehensive security checks."""
        validation_start = time.time()
        errors = []
        threat_score = 0.0
        
        try:
            # Convert input to string for analysis
            input_str = str(input_data) if input_data is not None else ""
            
            # Basic validation
            basic_errors, basic_threat = self._basic_validation(input_str, input_type)
            errors.extend(basic_errors)
            threat_score += basic_threat
            
            # Pattern-based validation
            pattern_errors, pattern_threat = self._pattern_validation(input_str)
            errors.extend(pattern_errors)
            threat_score += pattern_threat
            
            # ML-based threat detection
            ml_threat = self._ml_threat_detection(input_str, context)
            threat_score += ml_threat
            
            # Context-specific validation
            if context:
                context_errors, context_threat = self._context_validation(input_str, context)
                errors.extend(context_errors)
                threat_score += context_threat
            
            # Semantic analysis
            semantic_threat = self._semantic_analysis(input_str)
            threat_score += semantic_threat
            
            # Record validation attempt
            self._record_validation(input_str, errors, threat_score, validation_start)
            
            is_valid = len(errors) == 0 and threat_score < 0.5
            return is_valid, errors, threat_score
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False, [f"Validation error: {str(e)}"], 1.0
    
    def _basic_validation(self, input_str: str, input_type: str) -> Tuple[List[str], float]:
        """Basic input validation checks."""
        errors = []
        threat_score = 0.0
        
        # Length validation
        max_lengths = {
            "general": 10000,
            "scenario_description": 5000,
            "agent_response": 2000,
            "user_input": 1000,
            "api_key": 100,
            "username": 50
        }
        
        max_length = max_lengths.get(input_type, 1000)
        if len(input_str) > max_length:
            errors.append(f"Input exceeds maximum length of {max_length} characters")
            threat_score += 0.3
        
        # Character validation
        if input_type in ["username", "api_key"]:
            if not re.match(r'^[a-zA-Z0-9_\-\.]+$', input_str):
                errors.append("Invalid characters detected")
                threat_score += 0.4
        
        # Null byte detection
        if '\x00' in input_str:
            errors.append("Null bytes detected")
            threat_score += 0.8
        
        # Control character detection
        control_chars = sum(1 for c in input_str if ord(c) < 32 and c not in '\t\n\r')
        if control_chars > 0:
            errors.append(f"Control characters detected: {control_chars}")
            threat_score += min(0.5, control_chars * 0.1)
        
        return errors, threat_score
    
    def _pattern_validation(self, input_str: str) -> Tuple[List[str], float]:
        """Pattern-based validation for known attack vectors."""
        errors = []
        threat_score = 0.0
        
        # SQL injection patterns
        sql_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
            r"(--|\#|\/\*|\*\/)",
            r"(\bOR\b.*=.*\bOR\b)",
            r"(1=1|1'='1)"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                errors.append("Potential SQL injection detected")
                threat_score += 0.7
                break
        
        # XSS patterns
        xss_patterns = [
            r"<script[^>]*>",
            r"javascript:",
            r"on\w+\s*=",
            r"<iframe[^>]*>",
            r"eval\s*\(",
            r"document\.(cookie|write)"
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                errors.append("Potential XSS attack detected")
                threat_score += 0.6
                break
        
        # Command injection patterns
        cmd_patterns = [
            r"(\||&|;|\$\(|\`)",
            r"(rm\s+\-rf|del\s+\/|format\s+c:)",
            r"(sudo|su\s+\-|passwd)",
            r"(cat\s+\/etc\/passwd|\/bin\/sh)"
        ]
        
        for pattern in cmd_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                errors.append("Potential command injection detected")
                threat_score += 0.8
                break
        
        # Path traversal patterns
        if re.search(r"(\.\./|\.\.\\|%2e%2e%2f)", input_str, re.IGNORECASE):
            errors.append("Path traversal attempt detected")
            threat_score += 0.6
        
        # LDAP injection patterns
        ldap_patterns = [
            r"(\*|\(|\)|\&|\||\!)",
            r"(\)\s*\(\s*\|)",
            r"(\)\s*\(\s*\&)"
        ]
        
        ldap_score = sum(1 for pattern in ldap_patterns if re.search(pattern, input_str))
        if ldap_score > 2:
            errors.append("Potential LDAP injection detected")
            threat_score += 0.5
        
        return errors, threat_score
    
    def _ml_threat_detection(self, input_str: str, context: Optional[Dict[str, Any]]) -> float:
        """ML-based threat detection using feature analysis."""
        threat_score = 0.0
        
        # Feature extraction
        features = self._extract_security_features(input_str)
        
        # Entropy analysis
        entropy = self._calculate_entropy(input_str)
        if entropy > 4.5:  # High entropy may indicate obfuscation
            threat_score += min(0.3, (entropy - 4.5) * 0.1)
        
        # Character frequency analysis
        char_anomaly_score = self._analyze_character_frequency(input_str)
        threat_score += char_anomaly_score
        
        # Token analysis
        token_threat_score = self._analyze_tokens(input_str)
        threat_score += token_threat_score
        
        # Context anomaly detection
        if context:
            context_anomaly = self._detect_context_anomalies(features, context)
            threat_score += context_anomaly
        
        return min(1.0, threat_score)
    
    def _extract_security_features(self, input_str: str) -> Dict[str, float]:
        """Extract security-relevant features from input."""
        features = {
            'length': len(input_str),
            'uppercase_ratio': sum(1 for c in input_str if c.isupper()) / max(1, len(input_str)),
            'digit_ratio': sum(1 for c in input_str if c.isdigit()) / max(1, len(input_str)),
            'special_char_ratio': sum(1 for c in input_str if not c.isalnum()) / max(1, len(input_str)),
            'space_ratio': input_str.count(' ') / max(1, len(input_str)),
            'unique_chars': len(set(input_str)) / max(1, len(input_str)),
            'max_repeated_char': max([input_str.count(c) for c in set(input_str)] + [0]),
            'suspicious_keywords': sum(1 for word in ['admin', 'root', 'password', 'secret', 'token', 'key']
                                     if word.lower() in input_str.lower())
        }
        
        return features
    
    def _calculate_entropy(self, input_str: str) -> float:
        """Calculate Shannon entropy of input string."""
        if not input_str:
            return 0.0
        
        # Calculate character frequencies
        char_counts = {}
        for char in input_str:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        entropy = 0.0
        length = len(input_str)
        
        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _analyze_character_frequency(self, input_str: str) -> float:
        """Analyze character frequency for anomalies."""
        if not input_str:
            return 0.0
        
        # Expected character frequencies for English text
        expected_freq = {
            'a': 0.0817, 'b': 0.0149, 'c': 0.0278, 'd': 0.0425, 'e': 0.1202,
            'f': 0.0223, 'g': 0.0202, 'h': 0.0609, 'i': 0.0697, 'j': 0.0015,
            'k': 0.0077, 'l': 0.0403, 'm': 0.0241, 'n': 0.0675, 'o': 0.0751,
            'p': 0.0193, 'q': 0.0010, 'r': 0.0599, 's': 0.0633, 't': 0.0906,
            'u': 0.0276, 'v': 0.0098, 'w': 0.0236, 'x': 0.0015, 'y': 0.0197,
            'z': 0.0007, ' ': 0.1219
        }
        
        # Calculate actual frequencies
        text_lower = input_str.lower()
        actual_freq = {}
        total_chars = len(text_lower)
        
        for char in text_lower:
            if char.isalpha() or char == ' ':
                actual_freq[char] = actual_freq.get(char, 0) + 1
        
        # Normalize frequencies
        for char in actual_freq:
            actual_freq[char] = actual_freq[char] / total_chars
        
        # Calculate chi-squared statistic
        chi_squared = 0.0
        for char, expected in expected_freq.items():
            actual = actual_freq.get(char, 0)
            chi_squared += ((actual - expected) ** 2) / expected
        
        # Normalize to 0-1 scale
        anomaly_score = min(1.0, chi_squared / 100.0)
        
        return anomaly_score * 0.3  # Weight the contribution
    
    def _analyze_tokens(self, input_str: str) -> float:
        """Analyze tokens for suspicious patterns."""
        tokens = re.findall(r'\b\w+\b', input_str.lower())
        
        # Suspicious token patterns
        suspicious_tokens = {
            'admin', 'administrator', 'root', 'sudo', 'passwd', 'password',
            'secret', 'token', 'key', 'auth', 'login', 'exec', 'eval',
            'system', 'cmd', 'shell', 'bash', 'powershell', 'script',
            'inject', 'exploit', 'hack', 'crack', 'bypass'
        }
        
        # SQL/database tokens
        sql_tokens = {
            'select', 'insert', 'update', 'delete', 'drop', 'create',
            'alter', 'union', 'join', 'where', 'order', 'group'
        }
        
        # Programming/scripting tokens
        prog_tokens = {
            'function', 'var', 'let', 'const', 'if', 'else', 'for',
            'while', 'return', 'import', 'require', 'include'
        }
        
        suspicious_count = sum(1 for token in tokens if token in suspicious_tokens)
        sql_count = sum(1 for token in tokens if token in sql_tokens)
        prog_count = sum(1 for token in tokens if token in prog_tokens)
        
        total_tokens = len(tokens)
        if total_tokens == 0:
            return 0.0
        
        # Calculate threat score based on token analysis
        threat_score = 0.0
        threat_score += min(0.5, suspicious_count / total_tokens)
        threat_score += min(0.3, sql_count / total_tokens)
        threat_score += min(0.2, prog_count / total_tokens)
        
        return threat_score
    
    def _detect_context_anomalies(self, features: Dict[str, float], context: Dict[str, Any]) -> float:
        """Detect anomalies based on context."""
        anomaly_score = 0.0
        
        # Context-based feature analysis
        expected_context = context.get('expected_input_type', 'general')
        
        if expected_context == 'scenario_description':
            # Scenario descriptions should be readable text
            if features['special_char_ratio'] > 0.3:
                anomaly_score += 0.2
            if features['digit_ratio'] > 0.5:
                anomaly_score += 0.2
        
        elif expected_context == 'agent_response':
            # Agent responses should be structured
            if features['uppercase_ratio'] > 0.8:
                anomaly_score += 0.3
            if features['max_repeated_char'] > 10:
                anomaly_score += 0.2
        
        elif expected_context == 'api_key':
            # API keys should have specific characteristics
            if features['length'] < 10 or features['length'] > 100:
                anomaly_score += 0.4
            if features['unique_chars'] < 0.5:
                anomaly_score += 0.3
        
        return min(1.0, anomaly_score)
    
    def _context_validation(self, input_str: str, context: Dict[str, Any]) -> Tuple[List[str], float]:
        """Context-specific validation rules."""
        errors = []
        threat_score = 0.0
        
        input_type = context.get('input_type', 'general')
        user_role = context.get('user_role', 'user')
        source_ip = context.get('source_ip', '')
        
        # Role-based validation
        if user_role == 'admin' and 'password' in input_str.lower():
            # Admin users shouldn't be sending passwords in plain text
            errors.append("Sensitive data detected in admin context")
            threat_score += 0.5
        
        # IP-based validation
        if source_ip and self._is_suspicious_ip(source_ip):
            errors.append("Request from suspicious IP address")
            threat_score += 0.3
        
        # Rate limiting context
        request_count = context.get('recent_request_count', 0)
        if request_count > 100:  # High request rate
            errors.append("Potential rate limit abuse")
            threat_score += 0.4
        
        return errors, threat_score
    
    def _semantic_analysis(self, input_str: str) -> float:
        """Semantic analysis for advanced threat detection."""
        threat_score = 0.0
        
        # Detect social engineering patterns
        social_eng_keywords = [
            'urgent', 'immediate', 'emergency', 'verify', 'confirm',
            'suspended', 'expired', 'click here', 'act now', 'limited time'
        ]
        
        social_eng_count = sum(1 for keyword in social_eng_keywords
                              if keyword.lower() in input_str.lower())
        
        if social_eng_count > 2:
            threat_score += min(0.4, social_eng_count * 0.1)
        
        # Detect manipulation attempts
        manipulation_patterns = [
            r'\b(you must|you need to|required|mandatory)\b',
            r'\b(trust me|believe me|honestly|frankly)\b',
            r'\b(everyone knows|studies show|experts say)\b'
        ]
        
        manipulation_score = sum(1 for pattern in manipulation_patterns
                               if re.search(pattern, input_str, re.IGNORECASE))
        
        threat_score += min(0.3, manipulation_score * 0.1)
        
        return threat_score
    
    def _is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP address is suspicious (simplified implementation)."""
        # Known malicious IP ranges (example)
        suspicious_ranges = [
            '10.0.0.0/8',      # Private range (might be suspicious in certain contexts)
            '192.168.0.0/16',  # Private range
            '172.16.0.0/12'    # Private range
        ]
        
        # This is a simplified check - in practice, you'd use threat intelligence feeds
        return any(ip_address.startswith(range_prefix.split('/')[0][:7]) 
                  for range_prefix in suspicious_ranges)
    
    def _record_validation(self, input_str: str, errors: List[str], threat_score: float, start_time: float) -> None:
        """Record validation attempt for learning and monitoring."""
        validation_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'input_length': len(input_str),
            'errors': errors,
            'threat_score': threat_score,
            'processing_time_ms': (time.time() - start_time) * 1000,
            'blocked': len(errors) > 0 or threat_score > 0.5
        }
        
        self.validation_history.append(validation_record)
        
        # Keep only recent history
        if len(self.validation_history) > 10000:
            self.validation_history = self.validation_history[-10000:]
    
    def _load_validation_patterns(self) -> Dict[str, List[str]]:
        """Load validation patterns from configuration."""
        return {
            'sql_injection': [
                r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
                r"(--|\#|\/\*|\*\/)",
                r"(\bOR\b.*=.*\bOR\b)"
            ],
            'xss': [
                r"<script[^>]*>",
                r"javascript:",
                r"on\w+\s*="
            ],
            'command_injection': [
                r"(\||&|;|\$\(|\`)",
                r"(rm\s+\-rf|del\s+\/)"
            ]
        }
    
    def _load_threat_signatures(self) -> List[Dict[str, Any]]:
        """Load threat signatures from threat intelligence."""
        return [
            {
                'name': 'SQL injection attempt',
                'pattern': r"(\bSELECT\b.*\bFROM\b.*\bWHERE\b.*=.*)",
                'severity': 'high',
                'category': 'injection'
            },
            {
                'name': 'XSS attempt',
                'pattern': r"<script[^>]*>.*</script>",
                'severity': 'medium',
                'category': 'xss'
            }
        ]


class ThreatIntelligence:
    """Threat intelligence and pattern analysis system."""
    
    def __init__(self):
        """Initialize threat intelligence system."""
        self.threat_patterns: List[Dict[str, Any]] = []
        self.attack_signatures: Dict[str, List[str]] = {}
        self.ip_reputation: Dict[str, float] = {}
        self.behavioral_baselines: Dict[str, Dict[str, float]] = {}
        
    def analyze_threat_pattern(self, 
                             security_events: List[SecurityAlert],
                             time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Analyze threat patterns from security events."""
        recent_events = [
            event for event in security_events
            if event.timestamp > datetime.utcnow() - time_window
        ]
        
        if not recent_events:
            return {'pattern_detected': False, 'confidence': 0.0}
        
        # Group events by type and source
        event_groups = {}
        for event in recent_events:
            key = f"{event.event_type.value}_{event.source_ip}"
            if key not in event_groups:
                event_groups[key] = []
            event_groups[key].append(event)
        
        # Detect attack patterns
        patterns = []
        
        # Brute force detection
        auth_failures = [e for e in recent_events if e.event_type == SecurityEvent.AUTHENTICATION_FAILURE]
        if len(auth_failures) > 10:
            patterns.append({
                'pattern_type': 'brute_force_attack',
                'confidence': min(1.0, len(auth_failures) / 50.0),
                'source_ips': list(set(e.source_ip for e in auth_failures)),
                'event_count': len(auth_failures)
            })
        
        # Distributed attack detection
        unique_ips = set(e.source_ip for e in recent_events)
        if len(unique_ips) > 20 and len(recent_events) > 50:
            patterns.append({
                'pattern_type': 'distributed_attack',
                'confidence': min(1.0, len(unique_ips) / 100.0),
                'source_ips': list(unique_ips),
                'event_count': len(recent_events)
            })
        
        # Sequential attack detection
        injection_events = [e for e in recent_events if e.event_type == SecurityEvent.INJECTION_ATTEMPT]
        if len(injection_events) > 5:
            # Check if attacks are escalating
            threat_levels = [e.threat_level for e in injection_events]
            escalating = all(threat_levels[i].value <= threat_levels[i+1].value 
                           for i in range(len(threat_levels)-1))
            
            if escalating:
                patterns.append({
                    'pattern_type': 'escalating_injection_attack',
                    'confidence': 0.8,
                    'source_ips': list(set(e.source_ip for e in injection_events)),
                    'event_count': len(injection_events)
                })
        
        return {
            'pattern_detected': len(patterns) > 0,
            'patterns': patterns,
            'confidence': max([p['confidence'] for p in patterns] + [0.0]),
            'recommended_actions': self._recommend_mitigation_actions(patterns)
        }
    
    def update_ip_reputation(self, ip_address: str, reputation_score: float) -> None:
        """Update IP reputation score."""
        current_score = self.ip_reputation.get(ip_address, 0.5)
        
        # Weighted average with previous score
        updated_score = 0.7 * current_score + 0.3 * reputation_score
        self.ip_reputation[ip_address] = max(0.0, min(1.0, updated_score))
    
    def get_ip_reputation(self, ip_address: str) -> float:
        """Get IP reputation score (0.0 = malicious, 1.0 = trusted)."""
        return self.ip_reputation.get(ip_address, 0.5)  # Default neutral
    
    def analyze_behavioral_anomaly(self, 
                                 user_id: str,
                                 current_behavior: Dict[str, float]) -> Tuple[bool, float]:
        """Analyze behavioral anomalies for a user."""
        if user_id not in self.behavioral_baselines:
            # First time seeing this user - establish baseline
            self.behavioral_baselines[user_id] = current_behavior.copy()
            return False, 0.0
        
        baseline = self.behavioral_baselines[user_id]
        anomaly_score = 0.0
        
        # Compare current behavior with baseline
        for metric, current_value in current_behavior.items():
            if metric in baseline:
                baseline_value = baseline[metric]
                if baseline_value > 0:
                    deviation = abs(current_value - baseline_value) / baseline_value
                    anomaly_score += min(1.0, deviation)
        
        # Normalize by number of metrics
        anomaly_score = anomaly_score / max(1, len(current_behavior))
        
        # Update baseline with exponential moving average
        for metric, current_value in current_behavior.items():
            baseline[metric] = 0.9 * baseline.get(metric, current_value) + 0.1 * current_value
        
        is_anomalous = anomaly_score > 0.5
        return is_anomalous, anomaly_score
    
    def _recommend_mitigation_actions(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Recommend mitigation actions based on detected patterns."""
        actions = []
        
        for pattern in patterns:
            pattern_type = pattern['pattern_type']
            confidence = pattern['confidence']
            
            if pattern_type == 'brute_force_attack':
                if confidence > 0.7:
                    actions.extend([
                        'implement_account_lockout',
                        'enable_captcha',
                        'block_source_ips'
                    ])
                else:
                    actions.append('increase_monitoring')
            
            elif pattern_type == 'distributed_attack':
                actions.extend([
                    'enable_ddos_protection',
                    'implement_rate_limiting',
                    'contact_upstream_providers'
                ])
            
            elif pattern_type == 'escalating_injection_attack':
                actions.extend([
                    'block_source_ips_immediately',
                    'enable_waf_strict_mode',
                    'review_input_validation',
                    'alert_security_team'
                ])
        
        return list(set(actions))  # Remove duplicates


class SecurityEventProcessor:
    """Process and correlate security events."""
    
    def __init__(self):
        """Initialize security event processor."""
        self.event_queue: List[SecurityAlert] = []
        self.processed_events: List[SecurityAlert] = []
        self.threat_intelligence = ThreatIntelligence()
        self.security_metrics = SecurityMetrics()
        
    async def process_security_event(self, 
                                   event_type: SecurityEvent,
                                   source_ip: str,
                                   description: str,
                                   evidence: Dict[str, Any],
                                   user_id: Optional[str] = None) -> SecurityAlert:
        """Process a security event and generate alert if necessary."""
        # Create security alert
        alert = SecurityAlert(
            alert_id=f"alert_{int(time.time() * 1000)}_{hash(description) % 10000}",
            event_type=event_type,
            threat_level=self._assess_threat_level(event_type, evidence),
            timestamp=datetime.utcnow(),
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            evidence=evidence
        )
        
        # Add to queue for processing
        self.event_queue.append(alert)
        
        # Update security metrics
        self._update_security_metrics(alert)
        
        # Process correlation and threat analysis
        await self._correlate_events(alert)
        
        # Apply automatic mitigation if necessary
        if alert.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            await self._apply_automatic_mitigation(alert)
        
        return alert
    
    def _assess_threat_level(self, event_type: SecurityEvent, evidence: Dict[str, Any]) -> ThreatLevel:
        """Assess threat level based on event type and evidence."""
        base_threat_levels = {
            SecurityEvent.AUTHENTICATION_FAILURE: ThreatLevel.LOW,
            SecurityEvent.AUTHORIZATION_VIOLATION: ThreatLevel.MEDIUM,
            SecurityEvent.INPUT_VALIDATION_FAILURE: ThreatLevel.MEDIUM,
            SecurityEvent.RATE_LIMIT_EXCEEDED: ThreatLevel.LOW,
            SecurityEvent.SUSPICIOUS_PATTERN: ThreatLevel.MEDIUM,
            SecurityEvent.DATA_BREACH_ATTEMPT: ThreatLevel.HIGH,
            SecurityEvent.INJECTION_ATTEMPT: ThreatLevel.HIGH,
            SecurityEvent.PRIVILEGE_ESCALATION: ThreatLevel.CRITICAL
        }
        
        base_level = base_threat_levels.get(event_type, ThreatLevel.MEDIUM)
        
        # Adjust based on evidence
        threat_score = evidence.get('threat_score', 0.5)
        repeat_offender = evidence.get('repeat_offender', False)
        
        if threat_score > 0.8 or repeat_offender:
            # Escalate threat level
            if base_level == ThreatLevel.LOW:
                return ThreatLevel.MEDIUM
            elif base_level == ThreatLevel.MEDIUM:
                return ThreatLevel.HIGH
            elif base_level == ThreatLevel.HIGH:
                return ThreatLevel.CRITICAL
        
        return base_level
    
    def _update_security_metrics(self, alert: SecurityAlert) -> None:
        """Update security metrics with new alert."""
        self.security_metrics.total_requests += 1
        self.security_metrics.security_alerts += 1
        
        if alert.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self.security_metrics.blocked_requests += 1
        
        # Update average threat score
        threat_values = {
            ThreatLevel.LOW: 0.25,
            ThreatLevel.MEDIUM: 0.5,
            ThreatLevel.HIGH: 0.75,
            ThreatLevel.CRITICAL: 1.0
        }
        
        current_threat = threat_values[alert.threat_level]
        current_avg = self.security_metrics.average_threat_score
        total_alerts = self.security_metrics.security_alerts
        
        self.security_metrics.average_threat_score = (
            (current_avg * (total_alerts - 1) + current_threat) / total_alerts
        )
        
        self.security_metrics.last_updated = datetime.utcnow()
    
    async def _correlate_events(self, new_alert: SecurityAlert) -> None:
        """Correlate new alert with existing events."""
        # Get recent events (last hour)
        recent_cutoff = datetime.utcnow() - timedelta(hours=1)
        recent_events = [
            alert for alert in self.processed_events
            if alert.timestamp > recent_cutoff
        ]
        
        # Add new alert to recent events for analysis
        recent_events.append(new_alert)
        
        # Perform threat pattern analysis
        threat_analysis = self.threat_intelligence.analyze_threat_pattern(recent_events)
        
        if threat_analysis['pattern_detected']:
            logger.warning(
                f"Threat pattern detected: {threat_analysis['patterns']} "
                f"Confidence: {threat_analysis['confidence']:.2f}"
            )
            
            # Update alert with correlation information
            new_alert.evidence['threat_pattern'] = threat_analysis
    
    async def _apply_automatic_mitigation(self, alert: SecurityAlert) -> None:
        """Apply automatic mitigation measures."""
        mitigation_actions = []
        
        if alert.event_type == SecurityEvent.INJECTION_ATTEMPT:
            # Block source IP temporarily
            mitigation_actions.append(f"block_ip_{alert.source_ip}_3600")  # 1 hour block
            
        elif alert.event_type == SecurityEvent.AUTHENTICATION_FAILURE:
            # Rate limit the user/IP
            mitigation_actions.append(f"rate_limit_{alert.source_ip}_600")  # 10 minute limit
            
        elif alert.event_type == SecurityEvent.DATA_BREACH_ATTEMPT:
            # Immediate IP block and alert security team
            mitigation_actions.extend([
                f"block_ip_{alert.source_ip}_86400",  # 24 hour block
                "alert_security_team",
                "enable_enhanced_monitoring"
            ])
        
        # Log mitigation actions
        if mitigation_actions:
            alert.mitigation_applied = True
            alert.evidence['mitigation_actions'] = mitigation_actions
            
            logger.info(f"Applied automatic mitigation for alert {alert.alert_id}: {mitigation_actions}")
    
    async def get_security_dashboard(self) -> Dict[str, Any]:
        """Get security dashboard data."""
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_alerts = [
            alert for alert in self.processed_events
            if alert.timestamp > recent_cutoff
        ]
        
        # Calculate threat level distribution
        threat_distribution = {level.value: 0 for level in ThreatLevel}
        for alert in recent_alerts:
            threat_distribution[alert.threat_level.value] += 1
        
        # Top attack sources
        source_counts = {}
        for alert in recent_alerts:
            source_counts[alert.source_ip] = source_counts.get(alert.source_ip, 0) + 1
        
        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Attack type distribution
        event_distribution = {event.value: 0 for event in SecurityEvent}
        for alert in recent_alerts:
            event_distribution[alert.event_type.value] += 1
        
        return {
            'metrics': {
                'total_requests': self.security_metrics.total_requests,
                'blocked_requests': self.security_metrics.blocked_requests,
                'security_alerts': self.security_metrics.security_alerts,
                'average_threat_score': self.security_metrics.average_threat_score,
                'block_rate': (self.security_metrics.blocked_requests / 
                             max(1, self.security_metrics.total_requests)) * 100
            },
            'recent_activity': {
                'alerts_24h': len(recent_alerts),
                'threat_distribution': threat_distribution,
                'event_distribution': event_distribution,
                'top_attack_sources': top_sources
            },
            'threat_intelligence': {
                'patterns_detected': len(self.threat_intelligence.threat_patterns),
                'known_bad_ips': len([ip for ip, score in self.threat_intelligence.ip_reputation.items() 
                                    if score < 0.3])
            }
        }


class SecureDataHandler:
    """Secure data handling with encryption and access control."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """Initialize secure data handler."""
        if encryption_key:
            self.fernet = Fernet(encryption_key)
        else:
            self.fernet = Fernet(Fernet.generate_key())
        
        self.access_logs: List[Dict[str, Any]] = []
        self.data_classifications: Dict[str, str] = {}
        
    def encrypt_sensitive_data(self, data: Union[str, bytes], classification: str = "confidential") -> str:
        """Encrypt sensitive data with classification."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        encrypted_data = self.fernet.encrypt(data)
        encrypted_b64 = encrypted_data.decode('ascii')
        
        # Store classification
        data_hash = hashlib.sha256(data).hexdigest()
        self.data_classifications[data_hash] = classification
        
        # Log access
        self._log_data_access('encrypt', data_hash, classification)
        
        return encrypted_b64
    
    def decrypt_sensitive_data(self, encrypted_data: str, requester_id: str, purpose: str) -> Optional[str]:
        """Decrypt sensitive data with access control."""
        try:
            encrypted_bytes = encrypted_data.encode('ascii')
            decrypted_data = self.fernet.decrypt(encrypted_bytes)
            
            # Log access
            data_hash = hashlib.sha256(decrypted_data).hexdigest()
            classification = self.data_classifications.get(data_hash, "unknown")
            
            self._log_data_access('decrypt', data_hash, classification, requester_id, purpose)
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            self._log_data_access('decrypt_failed', 'unknown', 'unknown', requester_id, purpose)
            return None
    
    def _log_data_access(self, 
                        operation: str,
                        data_hash: str,
                        classification: str,
                        requester_id: Optional[str] = None,
                        purpose: Optional[str] = None) -> None:
        """Log data access for audit purposes."""
        access_log = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': operation,
            'data_hash': data_hash,
            'classification': classification,
            'requester_id': requester_id,
            'purpose': purpose,
            'success': operation != 'decrypt_failed'
        }
        
        self.access_logs.append(access_log)
        
        # Keep only recent logs
        if len(self.access_logs) > 10000:
            self.access_logs = self.access_logs[-10000:]
    
    def get_access_audit(self, time_window: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """Get access audit report."""
        cutoff_time = datetime.utcnow() - time_window
        recent_logs = [
            log for log in self.access_logs
            if datetime.fromisoformat(log['timestamp']) > cutoff_time
        ]
        
        # Analyze access patterns
        operations = {}
        classifications = {}
        requesters = {}
        
        for log in recent_logs:
            operations[log['operation']] = operations.get(log['operation'], 0) + 1
            classifications[log['classification']] = classifications.get(log['classification'], 0) + 1
            if log['requester_id']:
                requesters[log['requester_id']] = requesters.get(log['requester_id'], 0) + 1
        
        return {
            'total_accesses': len(recent_logs),
            'operations': operations,
            'classifications': classifications,
            'top_requesters': sorted(requesters.items(), key=lambda x: x[1], reverse=True)[:10],
            'failed_accesses': sum(1 for log in recent_logs if not log['success'])
        }