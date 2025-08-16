"""Global Optimization and Scaling Framework.

Multi-region deployment, I18n support, compliance framework,
and global-first implementation for Agent Skeptic Bench.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from geopy.distance import geodesic

logger = logging.getLogger(__name__)


class Region(Enum):
    """Global regions for deployment."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"


class ComplianceFramework(Enum):
    """Compliance frameworks."""
    GDPR = "gdpr"          # European Union
    CCPA = "ccpa"          # California
    PDPA = "pdpa"          # Singapore
    PIPEDA = "pipeda"      # Canada
    LGPD = "lgpd"          # Brazil


class Language(Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"


@dataclass
class RegionConfig:
    """Configuration for a specific region."""
    region: Region
    data_residency_required: bool
    compliance_frameworks: List[ComplianceFramework]
    preferred_languages: List[Language]
    latency_requirements_ms: int
    availability_sla: float  # 0.0 to 1.0
    cost_multiplier: float = 1.0


@dataclass
class GlobalMetrics:
    """Global deployment metrics."""
    total_regions: int
    active_regions: int
    global_latency_p95: float
    cross_region_sync_lag_ms: float
    compliance_coverage: Dict[str, bool]
    language_coverage: Dict[str, bool]
    data_residency_compliance: bool
    
    
class InternationalizationManager:
    """Internationalization and localization manager."""
    
    def __init__(self):
        """Initialize i18n manager."""
        self.translations: Dict[str, Dict[str, str]] = {}
        self.locale_configs: Dict[str, Dict[str, Any]] = {}
        self.supported_languages = set(lang.value for lang in Language)
        self._load_default_translations()
        
    def _load_default_translations(self) -> None:
        """Load default translations for core messages."""
        # Core skepticism evaluation messages
        core_messages = {
            "skepticism_high": {
                Language.ENGLISH.value: "High skepticism warranted for this claim",
                Language.SPANISH.value: "Se justifica un alto escepticismo para esta afirmación",
                Language.FRENCH.value: "Un scepticisme élevé est justifié pour cette affirmation",
                Language.GERMAN.value: "Hohe Skepsis ist bei dieser Behauptung angebracht",
                Language.JAPANESE.value: "この主張には高い懐疑心が必要です",
                Language.CHINESE.value: "对这个说法需要高度怀疑",
            },
            "evidence_insufficient": {
                Language.ENGLISH.value: "Insufficient evidence provided",
                Language.SPANISH.value: "Evidencia insuficiente proporcionada",
                Language.FRENCH.value: "Preuves insuffisantes fournies",
                Language.GERMAN.value: "Unzureichende Beweise vorgelegt",
                Language.JAPANESE.value: "提供された証拠が不十分です",
                Language.CHINESE.value: "提供的证据不足",
            },
            "claim_requires_verification": {
                Language.ENGLISH.value: "This claim requires independent verification",
                Language.SPANISH.value: "Esta afirmación requiere verificación independiente",
                Language.FRENCH.value: "Cette affirmation nécessite une vérification indépendante",
                Language.GERMAN.value: "Diese Behauptung erfordert eine unabhängige Überprüfung",
                Language.JAPANESE.value: "この主張は独立した検証が必要です",
                Language.CHINESE.value: "这个说法需要独立验证",
            },
            "confidence_level": {
                Language.ENGLISH.value: "Confidence Level",
                Language.SPANISH.value: "Nivel de Confianza",
                Language.FRENCH.value: "Niveau de Confiance",
                Language.GERMAN.value: "Vertrauensniveau",
                Language.JAPANESE.value: "信頼度",
                Language.CHINESE.value: "置信度",
            },
            "evaluation_complete": {
                Language.ENGLISH.value: "Evaluation completed successfully",
                Language.SPANISH.value: "Evaluación completada exitosamente",
                Language.FRENCH.value: "Évaluation terminée avec succès",
                Language.GERMAN.value: "Bewertung erfolgreich abgeschlossen",
                Language.JAPANESE.value: "評価が正常に完了しました",
                Language.CHINESE.value: "评估成功完成",
            }
        }
        
        for message_key, translations in core_messages.items():
            self.translations[message_key] = translations
    
    def get_message(self, key: str, language: str = "en", **kwargs) -> str:
        """Get localized message."""
        if key not in self.translations:
            return f"[MISSING: {key}]"
        
        if language not in self.translations[key]:
            # Fallback to English
            language = "en"
        
        if language not in self.translations[key]:
            return f"[MISSING: {key}:{language}]"
        
        message = self.translations[key][language]
        
        # Simple template substitution
        if kwargs:
            try:
                message = message.format(**kwargs)
            except KeyError:
                pass  # Ignore missing template variables
        
        return message
    
    def add_translation(self, key: str, language: str, message: str) -> None:
        """Add a translation."""
        if key not in self.translations:
            self.translations[key] = {}
        
        self.translations[key][language] = message
    
    def get_locale_config(self, language: str) -> Dict[str, Any]:
        """Get locale-specific configuration."""
        default_config = {
            "date_format": "%Y-%m-%d",
            "time_format": "%H:%M:%S",
            "decimal_separator": ".",
            "thousand_separator": ",",
            "currency_symbol": "$",
            "rtl": False  # Right-to-left text
        }
        
        locale_configs = {
            "en": default_config,
            "es": {**default_config, "currency_symbol": "€"},
            "fr": {**default_config, "decimal_separator": ",", "thousand_separator": " ", "currency_symbol": "€"},
            "de": {**default_config, "decimal_separator": ",", "thousand_separator": ".", "currency_symbol": "€"},
            "ja": {**default_config, "date_format": "%Y年%m月%d日", "currency_symbol": "¥"},
            "zh": {**default_config, "date_format": "%Y年%m月%d日", "currency_symbol": "¥"},
            "ar": {**default_config, "rtl": True, "currency_symbol": "﷼"},
        }
        
        return locale_configs.get(language, default_config)
    
    def format_number(self, number: float, language: str = "en") -> str:
        """Format number according to locale."""
        config = self.get_locale_config(language)
        
        # Simple formatting - in production, use proper locale libraries
        if config["decimal_separator"] == ",":
            formatted = f"{number:.2f}".replace(".", ",")
        else:
            formatted = f"{number:.2f}"
        
        return formatted
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.supported_languages)


class ComplianceManager:
    """Manage compliance with global data protection regulations."""
    
    def __init__(self):
        """Initialize compliance manager."""
        self.compliance_rules: Dict[ComplianceFramework, Dict[str, Any]] = {}
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.data_processing_logs: List[Dict[str, Any]] = []
        self._initialize_compliance_rules()
    
    def _initialize_compliance_rules(self) -> None:
        """Initialize compliance rules for different frameworks."""
        self.compliance_rules = {
            ComplianceFramework.GDPR: {
                "data_retention_days": 365,
                "requires_explicit_consent": True,
                "right_to_deletion": True,
                "right_to_portability": True,
                "data_protection_impact_assessment": True,
                "breach_notification_hours": 72,
                "pseudonymization_required": True,
                "allowed_transfer_countries": ["EU", "UK", "Switzerland"],
                "lawful_basis_required": True
            },
            ComplianceFramework.CCPA: {
                "data_retention_days": 365,
                "requires_explicit_consent": False,  # Opt-out model
                "right_to_deletion": True,
                "right_to_portability": True,
                "sale_opt_out": True,
                "disclosure_requirements": True,
                "third_party_sharing_disclosure": True
            },
            ComplianceFramework.PDPA: {
                "data_retention_days": 365,
                "requires_explicit_consent": True,
                "notification_requirements": True,
                "data_breach_notification": True,
                "cross_border_transfer_restrictions": True
            },
            ComplianceFramework.PIPEDA: {
                "data_retention_days": 365,
                "purpose_limitation": True,
                "consent_requirements": True,
                "access_rights": True,
                "breach_notification": True
            }
        }
    
    def check_compliance(self, 
                        framework: ComplianceFramework,
                        data_processing_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if data processing complies with framework."""
        if framework not in self.compliance_rules:
            return False, [f"Unknown compliance framework: {framework}"]
        
        rules = self.compliance_rules[framework]
        violations = []
        
        # Check data retention
        retention_days = data_processing_params.get("retention_days", 0)
        if retention_days > rules.get("data_retention_days", float('inf')):
            violations.append(f"Data retention exceeds maximum: {retention_days} > {rules['data_retention_days']}")
        
        # Check consent requirements
        if rules.get("requires_explicit_consent", False):
            if not data_processing_params.get("explicit_consent", False):
                violations.append("Explicit consent required but not provided")
        
        # Check lawful basis (GDPR)
        if framework == ComplianceFramework.GDPR:
            lawful_basis = data_processing_params.get("lawful_basis")
            valid_bases = ["consent", "contract", "legal_obligation", "vital_interests", "public_task", "legitimate_interests"]
            if not lawful_basis or lawful_basis not in valid_bases:
                violations.append(f"Valid lawful basis required, got: {lawful_basis}")
        
        # Check cross-border transfers
        transfer_country = data_processing_params.get("transfer_country")
        if transfer_country and framework == ComplianceFramework.GDPR:
            allowed_countries = rules.get("allowed_transfer_countries", [])
            if transfer_country not in allowed_countries:
                violations.append(f"Cross-border transfer to {transfer_country} not allowed")
        
        return len(violations) == 0, violations
    
    def record_consent(self, 
                      user_id: str,
                      purpose: str,
                      consent_given: bool,
                      framework: ComplianceFramework) -> None:
        """Record user consent."""
        consent_record = {
            "user_id": user_id,
            "purpose": purpose,
            "consent_given": consent_given,
            "framework": framework.value,
            "timestamp": datetime.utcnow().isoformat(),
            "consent_method": "explicit",
            "ip_address": "unknown",  # Should be provided in real implementation
            "user_agent": "unknown"   # Should be provided in real implementation
        }
        
        if user_id not in self.consent_records:
            self.consent_records[user_id] = {}
        
        self.consent_records[user_id][purpose] = consent_record
    
    def check_consent(self, user_id: str, purpose: str) -> bool:
        """Check if user has given consent for a purpose."""
        if user_id not in self.consent_records:
            return False
        
        if purpose not in self.consent_records[user_id]:
            return False
        
        return self.consent_records[user_id][purpose]["consent_given"]
    
    def process_data_deletion_request(self, user_id: str, framework: ComplianceFramework) -> Dict[str, Any]:
        """Process user's right to deletion request."""
        rules = self.compliance_rules.get(framework, {})
        
        if not rules.get("right_to_deletion", False):
            return {
                "success": False,
                "reason": f"Right to deletion not supported under {framework.value}"
            }
        
        # Log the deletion request
        deletion_log = {
            "user_id": user_id,
            "framework": framework.value,
            "requested_at": datetime.utcnow().isoformat(),
            "status": "processed",
            "data_categories": ["evaluation_results", "user_preferences", "consent_records"]
        }
        
        self.data_processing_logs.append(deletion_log)
        
        # Remove consent records
        if user_id in self.consent_records:
            del self.consent_records[user_id]
        
        return {
            "success": True,
            "deletion_reference": f"DEL_{int(time.time())}_{hash(user_id) % 10000}",
            "data_categories_deleted": deletion_log["data_categories"],
            "completion_time": datetime.utcnow().isoformat()
        }
    
    def generate_data_portability_export(self, user_id: str, framework: ComplianceFramework) -> Optional[Dict[str, Any]]:
        """Generate data export for user (right to portability)."""
        rules = self.compliance_rules.get(framework, {})
        
        if not rules.get("right_to_portability", False):
            return None
        
        export_data = {
            "user_id": user_id,
            "export_generated_at": datetime.utcnow().isoformat(),
            "framework": framework.value,
            "data_categories": {
                "consent_records": self.consent_records.get(user_id, {}),
                "processing_logs": [
                    log for log in self.data_processing_logs
                    if log.get("user_id") == user_id
                ]
            }
        }
        
        return export_data
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance status report."""
        report = {
            "supported_frameworks": [f.value for f in self.compliance_rules.keys()],
            "total_consent_records": len(self.consent_records),
            "total_processing_logs": len(self.data_processing_logs),
            "framework_coverage": {}
        }
        
        # Calculate framework-specific metrics
        for framework in self.compliance_rules.keys():
            framework_consents = sum(
                1 for user_consents in self.consent_records.values()
                for consent in user_consents.values()
                if consent["framework"] == framework.value
            )
            
            report["framework_coverage"][framework.value] = {
                "consent_records": framework_consents,
                "deletion_requests": sum(
                    1 for log in self.data_processing_logs
                    if log.get("framework") == framework.value and "deletion" in log.get("status", "")
                )
            }
        
        return report


class GlobalLoadBalancer:
    """Global load balancer for multi-region deployment."""
    
    def __init__(self):
        """Initialize global load balancer."""
        self.regions: Dict[Region, RegionConfig] = {}
        self.region_health: Dict[Region, float] = {}
        self.region_locations: Dict[Region, Tuple[float, float]] = {
            Region.US_EAST: (39.0458, -76.6413),      # Virginia
            Region.US_WEST: (45.5152, -122.6784),     # Oregon
            Region.EU_WEST: (53.4084, -8.2439),       # Ireland
            Region.EU_CENTRAL: (50.1109, 8.6821),     # Frankfurt
            Region.ASIA_PACIFIC: (1.3521, 103.8198),  # Singapore
            Region.ASIA_NORTHEAST: (35.6762, 139.6503) # Tokyo
        }
        self.routing_table: Dict[str, Region] = {}
        
    def add_region(self, config: RegionConfig) -> None:
        """Add a region to the load balancer."""
        self.regions[config.region] = config
        self.region_health[config.region] = 1.0  # Start with perfect health
        
    def update_region_health(self, region: Region, health_score: float) -> None:
        """Update health score for a region."""
        self.region_health[region] = max(0.0, min(1.0, health_score))
        
    def get_optimal_region(self, client_location: Optional[Tuple[float, float]] = None,
                          compliance_requirements: Optional[List[ComplianceFramework]] = None,
                          language_preference: Optional[Language] = None) -> Optional[Region]:
        """Get optimal region for a client request."""
        candidates = []
        
        for region, config in self.regions.items():
            health = self.region_health.get(region, 0.0)
            
            # Skip unhealthy regions
            if health < 0.5:
                continue
            
            # Check compliance requirements
            if compliance_requirements:
                if not any(framework in config.compliance_frameworks 
                          for framework in compliance_requirements):
                    continue
            
            # Calculate score
            score = health  # Base score is health
            
            # Add latency score
            if client_location:
                region_location = self.region_locations.get(region)
                if region_location:
                    distance_km = geodesic(client_location, region_location).kilometers
                    latency_score = max(0.0, 1.0 - (distance_km / 20000))  # Normalize by half Earth circumference
                    score += latency_score
            
            # Add language preference score
            if language_preference and language_preference in config.preferred_languages:
                score += 0.5
            
            # Add cost factor (lower cost = higher score)
            score += (2.0 - config.cost_multiplier) * 0.3
            
            candidates.append((region, score))
        
        if not candidates:
            return None
        
        # Return region with highest score
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def get_failover_regions(self, primary_region: Region, count: int = 2) -> List[Region]:
        """Get failover regions for a primary region."""
        candidates = []
        primary_config = self.regions.get(primary_region)
        
        if not primary_config:
            return []
        
        for region, config in self.regions.items():
            if region == primary_region:
                continue
            
            health = self.region_health.get(region, 0.0)
            if health < 0.3:  # Very low threshold for failover
                continue
            
            # Prefer regions with similar compliance frameworks
            compliance_overlap = len(set(primary_config.compliance_frameworks) & 
                                   set(config.compliance_frameworks))
            
            # Calculate distance penalty
            primary_location = self.region_locations.get(primary_region, (0, 0))
            region_location = self.region_locations.get(region, (0, 0))
            distance_km = geodesic(primary_location, region_location).kilometers
            distance_score = max(0.0, 1.0 - (distance_km / 20000))
            
            score = health + compliance_overlap * 0.3 + distance_score * 0.2
            candidates.append((region, score))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [region for region, _ in candidates[:count]]
    
    def route_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Route a request to the optimal region."""
        client_ip = request_context.get("client_ip")
        client_location = request_context.get("client_location")  # (lat, lon)
        compliance_requirements = request_context.get("compliance_requirements", [])
        language_preference = request_context.get("language_preference")
        
        # Convert language string to enum if needed
        if isinstance(language_preference, str):
            try:
                language_preference = Language(language_preference)
            except ValueError:
                language_preference = None
        
        optimal_region = self.get_optimal_region(
            client_location=client_location,
            compliance_requirements=compliance_requirements,
            language_preference=language_preference
        )
        
        if not optimal_region:
            return {
                "success": False,
                "error": "No suitable region available",
                "fallback_region": None
            }
        
        failover_regions = self.get_failover_regions(optimal_region)
        
        # Update routing table
        client_key = client_ip or "default"
        self.routing_table[client_key] = optimal_region
        
        return {
            "success": True,
            "primary_region": optimal_region.value,
            "failover_regions": [r.value for r in failover_regions],
            "estimated_latency_ms": self._estimate_latency(optimal_region, client_location),
            "compliance_frameworks": [f.value for f in self.regions[optimal_region].compliance_frameworks],
            "routing_decision_factors": {
                "health_score": self.region_health.get(optimal_region, 0.0),
                "compliance_match": len(compliance_requirements) > 0,
                "language_match": language_preference in self.regions[optimal_region].preferred_languages if language_preference else False
            }
        }
    
    def _estimate_latency(self, region: Region, client_location: Optional[Tuple[float, float]]) -> float:
        """Estimate latency to region."""
        if not client_location:
            return 100.0  # Default estimate
        
        region_location = self.region_locations.get(region, (0, 0))
        distance_km = geodesic(client_location, region_location).kilometers
        
        # Rough estimate: 1ms per 100km + base latency
        estimated_latency = (distance_km / 100) + 20
        return min(500, estimated_latency)  # Cap at 500ms
    
    def get_global_metrics(self) -> GlobalMetrics:
        """Get global deployment metrics."""
        total_regions = len(self.regions)
        active_regions = sum(1 for health in self.region_health.values() if health > 0.5)
        
        # Calculate global latency P95 (simplified)
        all_latencies = []
        for region in self.regions:
            for other_region in self.regions:
                if region != other_region:
                    region_loc = self.region_locations.get(region, (0, 0))
                    other_loc = self.region_locations.get(other_region, (0, 0))
                    latency = self._estimate_latency(other_region, region_loc)
                    all_latencies.append(latency)
        
        global_latency_p95 = np.percentile(all_latencies, 95) if all_latencies else 0
        
        # Calculate compliance coverage
        all_frameworks = set()
        for config in self.regions.values():
            all_frameworks.update(config.compliance_frameworks)
        
        compliance_coverage = {
            framework.value: any(framework in config.compliance_frameworks 
                               for config in self.regions.values())
            for framework in ComplianceFramework
        }
        
        # Calculate language coverage
        all_languages = set()
        for config in self.regions.values():
            all_languages.update(config.preferred_languages)
        
        language_coverage = {
            lang.value: any(lang in config.preferred_languages 
                          for config in self.regions.values())
            for lang in Language
        }
        
        return GlobalMetrics(
            total_regions=total_regions,
            active_regions=active_regions,
            global_latency_p95=global_latency_p95,
            cross_region_sync_lag_ms=50.0,  # Placeholder
            compliance_coverage=compliance_coverage,
            language_coverage=language_coverage,
            data_residency_compliance=all(config.data_residency_required 
                                        for config in self.regions.values())
        )


class CrossPlatformManager:
    """Cross-platform compatibility manager."""
    
    def __init__(self):
        """Initialize cross-platform manager."""
        self.platform_configs: Dict[str, Dict[str, Any]] = {}
        self.compatibility_matrix: Dict[str, Set[str]] = {}
        self._initialize_platform_configs()
    
    def _initialize_platform_configs(self) -> None:
        """Initialize platform-specific configurations."""
        self.platform_configs = {
            "web": {
                "supported_browsers": ["chrome", "firefox", "safari", "edge"],
                "min_browser_versions": {
                    "chrome": 80,
                    "firefox": 75,
                    "safari": 13,
                    "edge": 80
                },
                "required_features": ["es6", "fetch", "websockets"],
                "max_file_size_mb": 100,
                "supported_formats": ["json", "csv", "pdf"]
            },
            "mobile": {
                "supported_os": ["ios", "android"],
                "min_os_versions": {
                    "ios": "12.0",
                    "android": "7.0"
                },
                "required_permissions": ["internet", "storage"],
                "max_file_size_mb": 50,
                "supported_formats": ["json", "csv"]
            },
            "desktop": {
                "supported_os": ["windows", "macos", "linux"],
                "min_os_versions": {
                    "windows": "10",
                    "macos": "10.14",
                    "linux": "ubuntu-18.04"
                },
                "required_dependencies": ["python>=3.8", "nodejs>=14"],
                "max_file_size_mb": 500,
                "supported_formats": ["json", "csv", "pdf", "excel"]
            },
            "api": {
                "supported_protocols": ["http", "https", "websocket"],
                "authentication_methods": ["api_key", "oauth2", "jwt"],
                "rate_limits": {
                    "requests_per_minute": 1000,
                    "requests_per_hour": 10000
                },
                "max_payload_size_mb": 10,
                "supported_formats": ["json", "xml", "csv"]
            }
        }
        
        # Initialize compatibility matrix
        self.compatibility_matrix = {
            "web": {"desktop", "mobile"},
            "mobile": {"web"},
            "desktop": {"web", "api"},
            "api": {"web", "mobile", "desktop"}
        }
    
    def check_platform_compatibility(self, platform: str, requirements: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Check if platform meets requirements."""
        if platform not in self.platform_configs:
            return False, [f"Unknown platform: {platform}"]
        
        config = self.platform_configs[platform]
        issues = []
        
        # Check browser compatibility (web platform)
        if platform == "web" and "browser" in requirements:
            browser = requirements["browser"]
            version = requirements.get("browser_version", 0)
            
            if browser not in config["supported_browsers"]:
                issues.append(f"Browser {browser} not supported")
            elif version < config["min_browser_versions"].get(browser, 0):
                issues.append(f"Browser version {version} too old, minimum: {config['min_browser_versions'][browser]}")
        
        # Check OS compatibility (mobile/desktop)
        if platform in ["mobile", "desktop"] and "os" in requirements:
            os_name = requirements["os"]
            os_version = requirements.get("os_version", "0")
            
            if os_name not in config["supported_os"]:
                issues.append(f"OS {os_name} not supported")
            elif os_version < config["min_os_versions"].get(os_name, "0"):
                issues.append(f"OS version {os_version} too old, minimum: {config['min_os_versions'][os_name]}")
        
        # Check file size limits
        file_size_mb = requirements.get("file_size_mb", 0)
        if file_size_mb > config.get("max_file_size_mb", float('inf')):
            issues.append(f"File size {file_size_mb}MB exceeds limit: {config['max_file_size_mb']}MB")
        
        # Check format support
        requested_format = requirements.get("format")
        if requested_format and requested_format not in config.get("supported_formats", []):
            issues.append(f"Format {requested_format} not supported")
        
        return len(issues) == 0, issues
    
    def get_optimal_platform_config(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal platform configuration for user context."""
        user_agent = user_context.get("user_agent", "")
        device_type = user_context.get("device_type", "unknown")
        connection_speed = user_context.get("connection_speed", "unknown")
        
        # Determine platform from context
        if "mobile" in user_agent.lower() or device_type == "mobile":
            platform = "mobile"
        elif any(browser in user_agent.lower() for browser in ["chrome", "firefox", "safari", "edge"]):
            platform = "web"
        else:
            platform = "desktop"
        
        base_config = self.platform_configs.get(platform, {}).copy()
        
        # Adjust configuration based on connection speed
        if connection_speed in ["slow", "2g"]:
            # Reduce limits for slow connections
            base_config["max_file_size_mb"] = min(base_config.get("max_file_size_mb", 10), 10)
            base_config["optimization_level"] = "high"
        elif connection_speed in ["fast", "wifi", "5g"]:
            # Allow higher limits for fast connections
            base_config["optimization_level"] = "low"
        
        return {
            "detected_platform": platform,
            "recommended_config": base_config,
            "fallback_platforms": list(self.compatibility_matrix.get(platform, [])),
            "user_context": user_context
        }
    
    def get_cross_platform_report(self) -> Dict[str, Any]:
        """Generate cross-platform compatibility report."""
        total_platforms = len(self.platform_configs)
        
        # Calculate feature coverage across platforms
        all_features = set()
        for config in self.platform_configs.values():
            all_features.update(config.get("supported_formats", []))
            all_features.update(config.get("required_features", []))
        
        feature_coverage = {}
        for feature in all_features:
            supporting_platforms = []
            for platform, config in self.platform_configs.items():
                if (feature in config.get("supported_formats", []) or 
                    feature in config.get("required_features", [])):
                    supporting_platforms.append(platform)
            
            feature_coverage[feature] = {
                "supporting_platforms": supporting_platforms,
                "coverage_percentage": len(supporting_platforms) / total_platforms * 100
            }
        
        return {
            "total_platforms": total_platforms,
            "platform_list": list(self.platform_configs.keys()),
            "feature_coverage": feature_coverage,
            "compatibility_matrix": {
                platform: list(compatible) 
                for platform, compatible in self.compatibility_matrix.items()
            },
            "overall_compatibility_score": self._calculate_compatibility_score()
        }
    
    def _calculate_compatibility_score(self) -> float:
        """Calculate overall cross-platform compatibility score."""
        total_pairs = len(self.platform_configs) * (len(self.platform_configs) - 1)
        if total_pairs == 0:
            return 1.0
        
        compatible_pairs = sum(len(compatible) for compatible in self.compatibility_matrix.values())
        return compatible_pairs / total_pairs


class GlobalDeploymentManager:
    """Main manager for global deployment coordination."""
    
    def __init__(self):
        """Initialize global deployment manager."""
        self.i18n_manager = InternationalizationManager()
        self.compliance_manager = ComplianceManager()
        self.load_balancer = GlobalLoadBalancer()
        self.platform_manager = CrossPlatformManager()
        self.deployment_start_time = datetime.utcnow()
        
    def initialize_global_deployment(self, regions: List[RegionConfig]) -> Dict[str, Any]:
        """Initialize global deployment with specified regions."""
        results = {
            "initialized_regions": [],
            "failed_regions": [],
            "compliance_status": {},
            "i18n_status": {}
        }
        
        # Initialize regions
        for region_config in regions:
            try:
                self.load_balancer.add_region(region_config)
                results["initialized_regions"].append(region_config.region.value)
                
                # Check compliance for region
                compliance_status = {}
                for framework in region_config.compliance_frameworks:
                    is_compliant, violations = self.compliance_manager.check_compliance(
                        framework, 
                        {"retention_days": 365, "explicit_consent": True, "lawful_basis": "consent"}
                    )
                    compliance_status[framework.value] = {
                        "compliant": is_compliant,
                        "violations": violations
                    }
                
                results["compliance_status"][region_config.region.value] = compliance_status
                
            except Exception as e:
                results["failed_regions"].append({
                    "region": region_config.region.value,
                    "error": str(e)
                })
        
        # Initialize i18n for region languages
        all_languages = set()
        for region_config in regions:
            all_languages.update(lang.value for lang in region_config.preferred_languages)
        
        results["i18n_status"] = {
            "supported_languages": list(all_languages),
            "total_translations": len(self.i18n_manager.translations),
            "translation_coverage": self._calculate_translation_coverage(all_languages)
        }
        
        return results
    
    def _calculate_translation_coverage(self, required_languages: Set[str]) -> Dict[str, float]:
        """Calculate translation coverage for required languages."""
        coverage = {}
        
        for language in required_languages:
            total_keys = len(self.i18n_manager.translations)
            translated_keys = sum(
                1 for translations in self.i18n_manager.translations.values()
                if language in translations
            )
            
            coverage[language] = translated_keys / total_keys if total_keys > 0 else 0.0
        
        return coverage
    
    def process_global_request(self, request_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request with global optimization."""
        # Route request to optimal region
        routing_result = self.load_balancer.route_request(request_context)
        
        if not routing_result["success"]:
            return {
                "success": False,
                "error": "Failed to route request",
                "details": routing_result
            }
        
        # Get localized configuration
        language = request_context.get("language_preference", "en")
        platform_config = self.platform_manager.get_optimal_platform_config(request_context)
        
        # Check compliance requirements
        compliance_frameworks = request_context.get("compliance_requirements", [])
        compliance_checks = {}
        
        for framework_name in compliance_frameworks:
            try:
                framework = ComplianceFramework(framework_name)
                is_compliant, violations = self.compliance_manager.check_compliance(
                    framework,
                    request_context.get("data_processing_params", {})
                )
                compliance_checks[framework_name] = {
                    "compliant": is_compliant,
                    "violations": violations
                }
            except ValueError:
                compliance_checks[framework_name] = {
                    "compliant": False,
                    "violations": [f"Unknown compliance framework: {framework_name}"]
                }
        
        return {
            "success": True,
            "routing": routing_result,
            "localization": {
                "language": language,
                "locale_config": self.i18n_manager.get_locale_config(language),
                "platform_config": platform_config
            },
            "compliance": compliance_checks,
            "processing_region": routing_result["primary_region"],
            "estimated_latency_ms": routing_result["estimated_latency_ms"],
            "request_id": f"req_{int(time.time())}_{hash(str(request_context)) % 10000}"
        }
    
    def get_global_status_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive global status dashboard."""
        uptime_seconds = (datetime.utcnow() - self.deployment_start_time).total_seconds()
        
        # Get global metrics
        global_metrics = self.load_balancer.get_global_metrics()
        
        # Get compliance report
        compliance_report = self.compliance_manager.get_compliance_report()
        
        # Get cross-platform report
        platform_report = self.platform_manager.get_cross_platform_report()
        
        # Calculate overall health score
        health_factors = {
            "region_availability": global_metrics.active_regions / max(1, global_metrics.total_regions),
            "latency_performance": max(0, 1.0 - (global_metrics.global_latency_p95 / 1000)),  # Normalize by 1s
            "compliance_coverage": sum(global_metrics.compliance_coverage.values()) / max(1, len(global_metrics.compliance_coverage)),
            "platform_compatibility": platform_report["overall_compatibility_score"]
        }
        
        overall_health = sum(health_factors.values()) / len(health_factors)
        
        return {
            "overall_health_score": overall_health,
            "uptime_seconds": uptime_seconds,
            "global_metrics": {
                "total_regions": global_metrics.total_regions,
                "active_regions": global_metrics.active_regions,
                "global_latency_p95_ms": global_metrics.global_latency_p95,
                "compliance_frameworks_supported": len(compliance_report["supported_frameworks"]),
                "languages_supported": len(self.i18n_manager.get_supported_languages()),
                "platforms_supported": platform_report["total_platforms"]
            },
            "regional_status": {
                region.value: {
                    "health_score": self.load_balancer.region_health.get(region, 0.0),
                    "compliance_frameworks": [f.value for f in config.compliance_frameworks],
                    "preferred_languages": [l.value for l in config.preferred_languages]
                }
                for region, config in self.load_balancer.regions.items()
            },
            "compliance_status": compliance_report,
            "platform_status": platform_report,
            "health_factors": health_factors
        }
    
    def optimize_global_performance(self) -> Dict[str, Any]:
        """Optimize global performance based on current metrics."""
        optimizations = []
        
        # Analyze regional performance
        global_metrics = self.load_balancer.get_global_metrics()
        
        if global_metrics.global_latency_p95 > 500:  # High latency
            optimizations.append({
                "type": "latency_optimization",
                "action": "deploy_edge_caches",
                "estimated_improvement": "30-50% latency reduction"
            })
        
        if global_metrics.active_regions < global_metrics.total_regions:
            unhealthy_regions = global_metrics.total_regions - global_metrics.active_regions
            optimizations.append({
                "type": "availability_optimization",
                "action": f"investigate_{unhealthy_regions}_unhealthy_regions",
                "estimated_improvement": "increased global availability"
            })
        
        # Check compliance gaps
        compliance_gaps = [
            framework for framework, supported in global_metrics.compliance_coverage.items()
            if not supported
        ]
        
        if compliance_gaps:
            optimizations.append({
                "type": "compliance_optimization",
                "action": f"add_support_for_{compliance_gaps}",
                "estimated_improvement": "expanded market coverage"
            })
        
        # Check language gaps
        language_gaps = [
            language for language, supported in global_metrics.language_coverage.items()
            if not supported
        ]
        
        if language_gaps:
            optimizations.append({
                "type": "localization_optimization",
                "action": f"add_translations_for_{language_gaps[:3]}",  # Top 3 missing languages
                "estimated_improvement": "improved user experience in new markets"
            })
        
        return {
            "optimization_recommendations": optimizations,
            "current_performance": {
                "global_latency_p95": global_metrics.global_latency_p95,
                "region_availability": global_metrics.active_regions / max(1, global_metrics.total_regions),
                "compliance_coverage": sum(global_metrics.compliance_coverage.values()) / max(1, len(global_metrics.compliance_coverage)),
                "language_coverage": sum(global_metrics.language_coverage.values()) / max(1, len(global_metrics.language_coverage))
            },
            "optimization_priority": self._calculate_optimization_priority(optimizations)
        }
    
    def _calculate_optimization_priority(self, optimizations: List[Dict[str, Any]]) -> List[str]:
        """Calculate optimization priority based on impact."""
        priority_weights = {
            "availability_optimization": 1.0,  # Highest priority
            "latency_optimization": 0.8,
            "compliance_optimization": 0.6,
            "localization_optimization": 0.4
        }
        
        prioritized = sorted(
            optimizations,
            key=lambda opt: priority_weights.get(opt["type"], 0.0),
            reverse=True
        )
        
        return [opt["type"] for opt in prioritized]