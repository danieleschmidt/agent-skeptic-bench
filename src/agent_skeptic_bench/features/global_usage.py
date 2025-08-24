"""Global-first usage metrics with multi-region and i18n support."""

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Region(Enum):
    """Supported deployment regions."""
    
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"


class Language(Enum):
    """Supported languages for i18n."""
    
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"


class ComplianceFramework(Enum):
    """Data compliance frameworks."""
    
    GDPR = "gdpr"          # European General Data Protection Regulation
    CCPA = "ccpa"          # California Consumer Privacy Act
    PDPA = "pdpa"          # Personal Data Protection Act (Singapore/Thailand)
    LGPD = "lgpd"          # Lei Geral de Proteção de Dados (Brazil)


class GlobalUsageTracker:
    """Global usage tracker with multi-region and compliance support."""
    
    def __init__(self, 
                 region: Region = Region.US_EAST,
                 compliance_frameworks: List[ComplianceFramework] = None,
                 default_language: Language = Language.ENGLISH):
        """Initialize global usage tracker."""
        
        self.region = region
        self.compliance_frameworks = compliance_frameworks or [ComplianceFramework.GDPR]
        self.default_language = default_language
        
        # Region-specific configuration
        self.region_config = self._get_region_config(region)
        
        # Compliance configuration
        self.compliance_config = self._get_compliance_config(self.compliance_frameworks)
        
        # Storage paths (region-specific)
        self.storage_path = Path(f"data/usage_metrics/{region.value}")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"GlobalUsageTracker initialized: region={region.value}, compliance={[c.value for c in self.compliance_frameworks]}")
    
    def _get_region_config(self, region: Region) -> Dict[str, Any]:
        """Get region-specific configuration."""
        region_configs = {
            Region.US_EAST: {
                "timezone": "America/New_York",
                "data_residency": "us",
                "encryption_required": False,
                "retention_days": 365
            },
            Region.US_WEST: {
                "timezone": "America/Los_Angeles", 
                "data_residency": "us",
                "encryption_required": False,
                "retention_days": 365
            },
            Region.EU_WEST: {
                "timezone": "Europe/London",
                "data_residency": "eu",
                "encryption_required": True,  # GDPR requirement
                "retention_days": 90
            },
            Region.EU_CENTRAL: {
                "timezone": "Europe/Berlin",
                "data_residency": "eu", 
                "encryption_required": True,
                "retention_days": 90
            },
            Region.ASIA_PACIFIC: {
                "timezone": "Asia/Singapore",
                "data_residency": "apac",
                "encryption_required": True,
                "retention_days": 180
            },
            Region.ASIA_NORTHEAST: {
                "timezone": "Asia/Tokyo",
                "data_residency": "apac",
                "encryption_required": True,
                "retention_days": 180
            }
        }
        
        return region_configs.get(region, region_configs[Region.US_EAST])
    
    def _get_compliance_config(self, frameworks: List[ComplianceFramework]) -> Dict[str, Any]:
        """Get compliance configuration."""
        config = {
            "require_consent": False,
            "allow_profiling": True,
            "data_retention_days": 365,
            "require_encryption": False,
            "allow_cross_border_transfer": True,
            "require_audit_log": False
        }
        
        # Apply most restrictive requirements
        for framework in frameworks:
            if framework == ComplianceFramework.GDPR:
                config.update({
                    "require_consent": True,
                    "allow_profiling": False,  # Requires explicit consent
                    "data_retention_days": min(config["data_retention_days"], 90),
                    "require_encryption": True,
                    "allow_cross_border_transfer": False,  # Requires adequacy decision
                    "require_audit_log": True
                })
            
            elif framework == ComplianceFramework.CCPA:
                config.update({
                    "require_consent": True,
                    "data_retention_days": min(config["data_retention_days"], 365),
                    "require_audit_log": True
                })
            
            elif framework == ComplianceFramework.PDPA:
                config.update({
                    "require_consent": True,
                    "require_encryption": True,
                    "data_retention_days": min(config["data_retention_days"], 180),
                    "require_audit_log": True
                })
        
        return config
    
    def create_compliant_session(self, session_id: str, user_id: Optional[str] = None,
                               user_consent: bool = False, 
                               user_region: Optional[str] = None) -> Dict[str, Any]:
        """Create session with compliance validation."""
        
        # Check consent requirements
        if self.compliance_config["require_consent"] and not user_consent:
            return {
                "success": False,
                "error": "user_consent_required",
                "message": self._get_localized_message("consent_required", self.default_language)
            }
        
        # Check data residency requirements
        if user_region and not self._is_cross_border_allowed(user_region):
            return {
                "success": False,
                "error": "data_residency_violation", 
                "message": self._get_localized_message("data_residency_error", self.default_language)
            }
        
        # Create session with compliance metadata
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "consent_given": user_consent,
            "user_region": user_region,
            "data_region": self.region.value,
            "compliance_frameworks": [f.value for f in self.compliance_frameworks],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retention_until": self._calculate_retention_date()
        }
        
        # Log for audit trail if required
        if self.compliance_config["require_audit_log"]:
            self._log_audit_event("session_created", session_data)
        
        return {
            "success": True,
            "session_data": session_data,
            "compliance_info": {
                "frameworks_applied": [f.value for f in self.compliance_frameworks],
                "encryption_enabled": self.compliance_config["require_encryption"],
                "retention_days": self.compliance_config["data_retention_days"]
            }
        }
    
    def _is_cross_border_allowed(self, user_region: str) -> bool:
        """Check if cross-border data transfer is allowed."""
        if not self.compliance_config["allow_cross_border_transfer"]:
            # Map user region to data residency zones
            user_zone = self._get_data_residency_zone(user_region)
            current_zone = self.region_config["data_residency"]
            
            return user_zone == current_zone
        
        return True
    
    def _get_data_residency_zone(self, region: str) -> str:
        """Get data residency zone for a region."""
        region_mapping = {
            "us": "us", "canada": "us",
            "uk": "eu", "germany": "eu", "france": "eu", "spain": "eu", "italy": "eu",
            "singapore": "apac", "japan": "apac", "australia": "apac", "korea": "apac"
        }
        
        return region_mapping.get(region.lower(), "unknown")
    
    def _calculate_retention_date(self) -> str:
        """Calculate data retention expiration date."""
        from datetime import timedelta
        
        retention_days = self.compliance_config["data_retention_days"]
        retention_date = datetime.now(timezone.utc) + timedelta(days=retention_days)
        
        return retention_date.isoformat()
    
    def _log_audit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Log audit event for compliance."""
        audit_log = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "region": self.region.value,
            "compliance_frameworks": [f.value for f in self.compliance_frameworks],
            "event_data": event_data
        }
        
        # Write to audit log file
        audit_file = self.storage_path.parent / "audit" / f"audit_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_log, default=str) + "\n")
    
    def _get_localized_message(self, message_key: str, language: Language) -> str:
        """Get localized message for user communication."""
        messages = {
            "consent_required": {
                Language.ENGLISH: "User consent is required to process usage metrics.",
                Language.SPANISH: "Se requiere el consentimiento del usuario para procesar métricas de uso.",
                Language.FRENCH: "Le consentement de l'utilisateur est requis pour traiter les métriques d'utilisation.",
                Language.GERMAN: "Benutzerzustimmung ist erforderlich, um Nutzungsmetriken zu verarbeiten.",
                Language.JAPANESE: "使用メトリクスを処理するにはユーザーの同意が必要です。",
                Language.CHINESE: "处理使用指标需要用户同意。"
            },
            "data_residency_error": {
                Language.ENGLISH: "Data residency requirements prevent processing in this region.",
                Language.SPANISH: "Los requisitos de residencia de datos impiden el procesamiento en esta región.",
                Language.FRENCH: "Les exigences de résidence des données empêchent le traitement dans cette région.",
                Language.GERMAN: "Anforderungen zur Datenresidenz verhindern die Verarbeitung in dieser Region.",
                Language.JAPANESE: "データ常駐要件により、この地域での処理は禁止されています。",
                Language.CHINESE: "数据常驻要求阻止在此区域进行处理。"
            },
            "export_complete": {
                Language.ENGLISH: "Export completed successfully.",
                Language.SPANISH: "Exportación completada exitosamente.",
                Language.FRENCH: "Exportation terminée avec succès.",
                Language.GERMAN: "Export erfolgreich abgeschlossen.",
                Language.JAPANESE: "エクスポートが正常に完了しました。",
                Language.CHINESE: "导出成功完成。"
            }
        }
        
        return messages.get(message_key, {}).get(language, messages[message_key][Language.ENGLISH])


class MultiRegionSync:
    """Synchronizes usage metrics across multiple regions."""
    
    def __init__(self, regions: List[Region]):
        """Initialize multi-region sync."""
        self.regions = regions
        self.sync_conflicts = []
        self.last_sync = {}
        
        logger.info(f"MultiRegionSync initialized for regions: {[r.value for r in regions]}")
    
    async def sync_usage_data(self, sync_window_hours: int = 1) -> Dict[str, Any]:
        """Synchronize usage data across regions."""
        sync_results = {}
        
        for region in self.regions:
            try:
                # Get regional data (simulated)
                regional_data = await self._get_regional_data(region, sync_window_hours)
                
                # Check for conflicts
                conflicts = self._detect_conflicts(region, regional_data)
                
                sync_results[region.value] = {
                    "success": True,
                    "records_synced": len(regional_data),
                    "conflicts": len(conflicts),
                    "last_sync": datetime.now(timezone.utc).isoformat()
                }
                
                if conflicts:
                    self.sync_conflicts.extend(conflicts)
                
                self.last_sync[region.value] = datetime.now(timezone.utc)
                
            except Exception as e:
                sync_results[region.value] = {
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"Sync failed for region {region.value}: {e}")
        
        return {
            "sync_timestamp": datetime.now(timezone.utc).isoformat(),
            "regions_synced": len([r for r in sync_results.values() if r.get("success")]),
            "total_conflicts": len(self.sync_conflicts),
            "regional_results": sync_results
        }
    
    async def _get_regional_data(self, region: Region, hours: int) -> List[Dict[str, Any]]:
        """Get usage data from a specific region."""
        # Simulate regional data retrieval
        import random
        
        regional_data = []
        for i in range(random.randint(50, 200)):  # Varying data volume per region
            regional_data.append({
                "session_id": f"{region.value}_session_{i:04d}",
                "region": region.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "evaluations": random.randint(1, 10),
                "score": random.uniform(0.6, 0.95)
            })
        
        return regional_data
    
    def _detect_conflicts(self, region: Region, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect synchronization conflicts."""
        conflicts = []
        
        # Simple conflict detection (same session ID in different regions)
        for record in data:
            session_id = record.get("session_id", "")
            
            # Check if session ID appears to be from another region
            for other_region in self.regions:
                if other_region != region and other_region.value in session_id:
                    conflicts.append({
                        "type": "cross_region_session",
                        "session_id": session_id,
                        "expected_region": region.value,
                        "actual_region": other_region.value,
                        "timestamp": record.get("timestamp")
                    })
        
        return conflicts


class ComplianceManager:
    """Manages data compliance across different frameworks."""
    
    def __init__(self, frameworks: List[ComplianceFramework]):
        """Initialize compliance manager."""
        self.frameworks = frameworks
        self.compliance_rules = self._load_compliance_rules()
        
        logger.info(f"ComplianceManager initialized for: {[f.value for f in frameworks]}")
    
    def _load_compliance_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load compliance rules for each framework."""
        return {
            ComplianceFramework.GDPR.value: {
                "max_retention_days": 90,
                "require_encryption": True,
                "require_consent": True,
                "allow_profiling": False,
                "require_data_portability": True,
                "require_deletion_capability": True,
                "lawful_basis_required": True
            },
            ComplianceFramework.CCPA.value: {
                "max_retention_days": 365,
                "require_encryption": False,
                "require_consent": True,
                "allow_profiling": True,
                "require_data_portability": True,
                "require_deletion_capability": True,
                "lawful_basis_required": False
            },
            ComplianceFramework.PDPA.value: {
                "max_retention_days": 180,
                "require_encryption": True,
                "require_consent": True,
                "allow_profiling": False,
                "require_data_portability": True,
                "require_deletion_capability": True,
                "lawful_basis_required": True
            },
            ComplianceFramework.LGPD.value: {
                "max_retention_days": 180,
                "require_encryption": True,
                "require_consent": True,
                "allow_profiling": False,
                "require_data_portability": True,
                "require_deletion_capability": True,
                "lawful_basis_required": True
            }
        }
    
    def validate_data_processing(self, user_data: Dict[str, Any], 
                                processing_purpose: str) -> Dict[str, Any]:
        """Validate data processing against compliance requirements."""
        validation_result = {
            "compliant": True,
            "violations": [],
            "requirements": [],
            "recommendations": []
        }
        
        for framework in self.frameworks:
            rules = self.compliance_rules[framework.value]
            
            # Check consent
            if rules["require_consent"] and not user_data.get("consent_given", False):
                validation_result["violations"].append({
                    "framework": framework.value,
                    "violation": "missing_consent",
                    "message": "User consent required for data processing"
                })
                validation_result["compliant"] = False
            
            # Check retention period
            if "retention_until" in user_data:
                retention_date = datetime.fromisoformat(user_data["retention_until"])
                max_retention = datetime.now(timezone.utc) + timedelta(days=rules["max_retention_days"])
                
                if retention_date > max_retention:
                    validation_result["violations"].append({
                        "framework": framework.value,
                        "violation": "excessive_retention",
                        "message": f"Retention period exceeds {rules['max_retention_days']} days"
                    })
                    validation_result["compliant"] = False
            
            # Check encryption requirements
            if rules["require_encryption"] and not user_data.get("encrypted", False):
                validation_result["requirements"].append({
                    "framework": framework.value,
                    "requirement": "encryption_required",
                    "message": "Data must be encrypted for compliance"
                })
        
        return validation_result
    
    def get_user_rights(self, framework: ComplianceFramework, user_region: str) -> List[str]:
        """Get user rights under a specific compliance framework."""
        rights_mapping = {
            ComplianceFramework.GDPR: [
                "access",           # Right to access personal data
                "rectification",    # Right to correct inaccurate data
                "erasure",          # Right to be forgotten
                "portability",      # Right to data portability
                "restriction",      # Right to restrict processing
                "objection"         # Right to object to processing
            ],
            ComplianceFramework.CCPA: [
                "access",           # Right to know what data is collected
                "deletion",         # Right to delete personal information
                "portability",      # Right to data portability
                "opt_out"           # Right to opt out of sale
            ],
            ComplianceFramework.PDPA: [
                "access",           # Right to access personal data
                "correction",       # Right to correct personal data
                "deletion",         # Right to delete personal data
                "portability"       # Right to data portability
            ]
        }
        
        return rights_mapping.get(framework, [])


class InternationalizationManager:
    """Manages internationalization for usage metrics UI and messages."""
    
    def __init__(self, default_language: Language = Language.ENGLISH):
        """Initialize i18n manager."""
        self.default_language = default_language
        self.translations = self._load_translations()
        
        logger.info(f"InternationalizationManager initialized: default={default_language.value}")
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation strings."""
        return {
            "usage_summary_title": {
                "en": "Usage Summary",
                "es": "Resumen de Uso", 
                "fr": "Résumé d'Utilisation",
                "de": "Nutzungsübersicht",
                "ja": "使用状況の概要",
                "zh": "使用摘要"
            },
            "total_sessions": {
                "en": "Total Sessions",
                "es": "Sesiones Totales",
                "fr": "Sessions Totales", 
                "de": "Gesamtsitzungen",
                "ja": "総セッション数",
                "zh": "总会话数"
            },
            "evaluation_count": {
                "en": "Evaluations",
                "es": "Evaluaciones",
                "fr": "Évaluations",
                "de": "Bewertungen", 
                "ja": "評価数",
                "zh": "评估次数"
            },
            "average_score": {
                "en": "Average Score",
                "es": "Puntuación Promedio",
                "fr": "Score Moyen",
                "de": "Durchschnittlicher Score",
                "ja": "平均スコア", 
                "zh": "平均分数"
            },
            "export_formats": {
                "en": "Available export formats: JSON, CSV, Excel",
                "es": "Formatos de exportación disponibles: JSON, CSV, Excel",
                "fr": "Formats d'exportation disponibles : JSON, CSV, Excel",
                "de": "Verfügbare Exportformate: JSON, CSV, Excel",
                "ja": "利用可能なエクスポート形式: JSON、CSV、Excel",
                "zh": "可用导出格式：JSON、CSV、Excel"
            }
        }
    
    def get_translated_text(self, key: str, language: Language = None) -> str:
        """Get translated text for a key."""
        target_language = language or self.default_language
        
        if key not in self.translations:
            logger.warning(f"Translation key not found: {key}")
            return key
        
        translations = self.translations[key]
        
        return translations.get(target_language.value, translations.get("en", key))
    
    def get_localized_export_config(self, language: Language) -> Dict[str, Any]:
        """Get localized export configuration."""
        return {
            "language": language.value,
            "date_format": self._get_date_format(language),
            "number_format": self._get_number_format(language),
            "currency_format": self._get_currency_format(language),
            "translations": {
                key: self.get_translated_text(key, language)
                for key in ["usage_summary_title", "total_sessions", "evaluation_count", "average_score"]
            }
        }
    
    def _get_date_format(self, language: Language) -> str:
        """Get date format for language."""
        formats = {
            Language.ENGLISH: "%Y-%m-%d %H:%M:%S",
            Language.SPANISH: "%d/%m/%Y %H:%M:%S",
            Language.FRENCH: "%d/%m/%Y %H:%M:%S", 
            Language.GERMAN: "%d.%m.%Y %H:%M:%S",
            Language.JAPANESE: "%Y年%m月%d日 %H:%M:%S",
            Language.CHINESE: "%Y年%m月%d日 %H:%M:%S"
        }
        
        return formats.get(language, formats[Language.ENGLISH])
    
    def _get_number_format(self, language: Language) -> Dict[str, str]:
        """Get number formatting for language."""
        formats = {
            Language.ENGLISH: {"decimal": ".", "thousands": ","},
            Language.SPANISH: {"decimal": ",", "thousands": "."},
            Language.FRENCH: {"decimal": ",", "thousands": " "},
            Language.GERMAN: {"decimal": ",", "thousands": "."},
            Language.JAPANESE: {"decimal": ".", "thousands": ","},
            Language.CHINESE: {"decimal": ".", "thousands": ","}
        }
        
        return formats.get(language, formats[Language.ENGLISH])
    
    def _get_currency_format(self, language: Language) -> str:
        """Get currency format for language."""
        formats = {
            Language.ENGLISH: "${amount}",
            Language.SPANISH: "{amount} €",
            Language.FRENCH: "{amount} €",
            Language.GERMAN: "{amount} €",
            Language.JAPANESE: "¥{amount}",
            Language.CHINESE: "¥{amount}"
        }
        
        return formats.get(language, formats[Language.ENGLISH])


class GlobalExportManager:
    """Manages exports with global compliance and localization."""
    
    def __init__(self, region: Region, compliance_manager: ComplianceManager,
                 i18n_manager: InternationalizationManager):
        """Initialize global export manager."""
        self.region = region
        self.compliance_manager = compliance_manager
        self.i18n_manager = i18n_manager
        
        logger.info(f"GlobalExportManager initialized for region {region.value}")
    
    async def export_with_compliance(self, 
                                   export_params: Dict[str, Any],
                                   user_language: Language = Language.ENGLISH) -> Dict[str, Any]:
        """Export data with compliance and localization."""
        
        # Validate compliance
        compliance_check = self.compliance_manager.validate_data_processing(
            export_params, "data_export"
        )
        
        if not compliance_check["compliant"]:
            return {
                "success": False,
                "compliance_violations": compliance_check["violations"],
                "message": self.i18n_manager.get_translated_text("compliance_violation", user_language)
            }
        
        # Get localized export config
        export_config = self.i18n_manager.get_localized_export_config(user_language)
        
        # Simulate export process
        export_data = {
            "export_info": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "region": self.region.value,
                "language": user_language.value,
                "compliance_frameworks": [f.value for f in self.compliance_manager.frameworks],
                "record_count": export_params.get("record_count", 0)
            },
            "localized_headers": export_config["translations"],
            "data": export_params.get("data", [])
        }
        
        # Apply data masking if required
        if any(f == ComplianceFramework.GDPR for f in self.compliance_manager.frameworks):
            export_data = self._apply_data_masking(export_data)
        
        return {
            "success": True,
            "export_data": export_data,
            "compliance_info": compliance_check,
            "localization": export_config,
            "message": self.i18n_manager.get_translated_text("export_complete", user_language)
        }
    
    def _apply_data_masking(self, export_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply data masking for compliance."""
        # Mask sensitive fields for GDPR compliance
        if "data" in export_data:
            for record in export_data["data"]:
                if "user_id" in record:
                    # Hash user ID for privacy
                    import hashlib
                    user_id = record["user_id"]
                    if user_id:
                        hashed_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
                        record["user_id_hash"] = hashed_id
                        del record["user_id"]
        
        return export_data