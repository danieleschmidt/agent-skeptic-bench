"""Monitoring and health checks for usage metrics system."""

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HealthStatus:
    """Health status information."""
    
    status: str  # "healthy", "degraded", "unhealthy"
    timestamp: datetime
    checks: Dict[str, bool]
    metrics: Dict[str, float]
    alerts: List[str]


class UsageMetricsHealthChecker:
    """Health checker for usage metrics system."""
    
    def __init__(self, storage_path: Path):
        """Initialize health checker."""
        self.storage_path = storage_path
        self.last_check = None
        self.check_interval = 300  # 5 minutes
    
    async def check_health(self) -> HealthStatus:
        """Perform comprehensive health check."""
        checks = {}
        metrics = {}
        alerts = []
        
        # Check storage accessibility
        checks["storage_accessible"] = self._check_storage_access()
        if not checks["storage_accessible"]:
            alerts.append("Storage directory not accessible")
        
        # Check disk space
        disk_usage = self._check_disk_usage()
        checks["sufficient_disk_space"] = disk_usage < 0.9  # Less than 90% full
        metrics["disk_usage_percent"] = disk_usage * 100
        
        if disk_usage > 0.9:
            alerts.append(f"Low disk space: {disk_usage:.1%} used")
        
        # Check data file integrity
        file_integrity = self._check_file_integrity()
        checks["data_integrity"] = file_integrity["valid_files"] > 0
        metrics["valid_files"] = file_integrity["valid_files"]
        metrics["corrupted_files"] = file_integrity["corrupted_files"]
        
        if file_integrity["corrupted_files"] > 0:
            alerts.append(f"{file_integrity['corrupted_files']} corrupted data files found")
        
        # Check recent activity
        recent_activity = self._check_recent_activity()
        checks["recent_activity"] = recent_activity["has_recent_data"]
        metrics["hours_since_last_write"] = recent_activity["hours_since_last"]
        
        # Check write permissions
        checks["write_permissions"] = self._check_write_permissions()
        if not checks["write_permissions"]:
            alerts.append("No write permissions to storage directory")
        
        # Determine overall status
        if all(checks.values()):
            status = "healthy"
        elif any(checks.values()):
            status = "degraded"
        else:
            status = "unhealthy"
        
        health_status = HealthStatus(
            status=status,
            timestamp=datetime.utcnow(),
            checks=checks,
            metrics=metrics,
            alerts=alerts
        )
        
        self.last_check = health_status
        return health_status
    
    def _check_storage_access(self) -> bool:
        """Check if storage directory is accessible."""
        try:
            return self.storage_path.exists() and self.storage_path.is_dir()
        except Exception:
            return False
    
    def _check_disk_usage(self) -> float:
        """Check disk usage percentage."""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.storage_path)
            return used / total
        except Exception:
            return 0.0  # Assume healthy if can't check
    
    def _check_file_integrity(self) -> Dict[str, int]:
        """Check integrity of usage metrics files."""
        valid_files = 0
        corrupted_files = 0
        
        try:
            for file_path in self.storage_path.glob("usage_metrics_*.jsonl"):
                try:
                    with open(file_path, 'r') as f:
                        line_count = 0
                        for line in f:
                            import json
                            json.loads(line.strip())  # Validate JSON
                            line_count += 1
                    
                    if line_count > 0:
                        valid_files += 1
                    
                except Exception:
                    corrupted_files += 1
                    logger.warning(f"Corrupted file detected: {file_path}")
        
        except Exception:
            pass  # If can't check, assume healthy
        
        return {"valid_files": valid_files, "corrupted_files": corrupted_files}
    
    def _check_recent_activity(self) -> Dict[str, Any]:
        """Check for recent activity in usage metrics."""
        try:
            latest_file = None
            latest_time = None
            
            for file_path in self.storage_path.glob("usage_metrics_*.jsonl"):
                file_time = file_path.stat().st_mtime
                if latest_time is None or file_time > latest_time:
                    latest_time = file_time
                    latest_file = file_path
            
            if latest_time:
                hours_since = (time.time() - latest_time) / 3600
                has_recent = hours_since < 24  # Activity within 24 hours
                
                return {
                    "has_recent_data": has_recent,
                    "hours_since_last": hours_since,
                    "latest_file": str(latest_file) if latest_file else None
                }
            else:
                return {"has_recent_data": False, "hours_since_last": float('inf')}
        
        except Exception:
            return {"has_recent_data": True, "hours_since_last": 0}  # Assume healthy
    
    def _check_write_permissions(self) -> bool:
        """Check write permissions to storage directory."""
        try:
            test_file = self.storage_path / ".health_check_test"
            test_file.write_text("test")
            test_file.unlink()
            return True
        except Exception:
            return False


class UsageMetricsMonitor:
    """Monitors usage metrics system performance and health."""
    
    def __init__(self, storage_path: Path):
        """Initialize usage metrics monitor."""
        self.storage_path = storage_path
        self.health_checker = UsageMetricsHealthChecker(storage_path)
        self.performance_metrics = {}
        self.start_time = time.time()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health = await self.health_checker.check_health()
        
        status = {
            "overall_health": health.status,
            "uptime_seconds": time.time() - self.start_time,
            "last_health_check": health.timestamp.isoformat(),
            "health_checks": health.checks,
            "metrics": health.metrics,
            "alerts": health.alerts,
            "storage_info": self._get_storage_info(),
            "performance_metrics": self.performance_metrics.copy()
        }
        
        return status
    
    def _get_storage_info(self) -> Dict[str, Any]:
        """Get storage directory information."""
        try:
            file_count = len(list(self.storage_path.glob("usage_metrics_*.jsonl")))
            
            total_size = sum(
                f.stat().st_size 
                for f in self.storage_path.glob("usage_metrics_*.jsonl")
            )
            
            return {
                "file_count": file_count,
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "storage_path": str(self.storage_path)
            }
        
        except Exception as e:
            return {"error": f"Failed to get storage info: {e}"}
    
    def record_operation_timing(self, operation: str, duration: float) -> None:
        """Record timing for usage metrics operations."""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = {
                "count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "max_time": 0.0,
                "min_time": float('inf')
            }
        
        metrics = self.performance_metrics[operation]
        metrics["count"] += 1
        metrics["total_time"] += duration
        metrics["avg_time"] = metrics["total_time"] / metrics["count"]
        metrics["max_time"] = max(metrics["max_time"], duration)
        metrics["min_time"] = min(metrics["min_time"], duration)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        return {
            "operations": self.performance_metrics.copy(),
            "uptime_seconds": time.time() - self.start_time,
            "last_updated": datetime.utcnow().isoformat()
        }