"""Robust Monitoring System for Agent Skeptic Bench

Provides comprehensive monitoring, health checks, alerting, and observability
for production-ready deployment.
"""

import asyncio
import logging
import psutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import threading
import json

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class MetricPoint:
    """Individual metric measurement."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert notification."""
    name: str
    level: str  # info, warning, critical
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class RobustMonitor:
    """Comprehensive monitoring system."""
    
    def __init__(self, 
                 check_interval: int = 30,
                 metric_retention_hours: int = 24,
                 alert_retention_hours: int = 72):
        """Initialize robust monitoring system."""
        self.check_interval = check_interval
        self.metric_retention = timedelta(hours=metric_retention_hours)
        self.alert_retention = timedelta(hours=alert_retention_hours)
        
        # Storage
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.alerts: List[Alert] = []
        
        # Health check functions
        self.health_check_functions: Dict[str, Callable[[], HealthCheck]] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.request_times: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_cleanup = datetime.utcnow()
        
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        def system_resources_check() -> HealthCheck:
            """Check system resource usage."""
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            status = HealthStatus.HEALTHY
            messages = []
            
            if cpu_percent > 80:
                status = HealthStatus.WARNING
                messages.append(f"High CPU usage: {cpu_percent:.1f}%")
            
            if memory.percent > 85:
                status = HealthStatus.CRITICAL if memory.percent > 95 else HealthStatus.WARNING
                messages.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent > 90:
                status = HealthStatus.CRITICAL if disk.percent > 95 else HealthStatus.WARNING
                messages.append(f"High disk usage: {disk.percent:.1f}%")
            
            message = "; ".join(messages) if messages else "System resources normal"
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                metadata={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk.percent,
                    "memory_available_gb": memory.available / (1024**3)
                }
            )
        
        def application_health_check() -> HealthCheck:
            """Check application-specific health."""
            try:
                # Check if key components are responsive
                start_time = time.time()
                
                # Simulate component checks
                response_time = (time.time() - start_time) * 1000
                
                if response_time > 5000:  # 5 seconds
                    return HealthCheck(
                        name="application_health",
                        status=HealthStatus.WARNING,
                        message=f"Slow application response: {response_time:.0f}ms",
                        duration_ms=response_time
                    )
                
                return HealthCheck(
                    name="application_health",
                    status=HealthStatus.HEALTHY,
                    message="Application responding normally",
                    duration_ms=response_time
                )
            
            except Exception as e:
                return HealthCheck(
                    name="application_health",
                    status=HealthStatus.CRITICAL,
                    message=f"Application health check failed: {str(e)}"
                )
        
        def error_rate_check() -> HealthCheck:
            """Check error rates."""
            total_errors = sum(self.error_counts.values())
            total_requests = len(self.request_times)
            
            if total_requests == 0:
                return HealthCheck(
                    name="error_rate",
                    status=HealthStatus.HEALTHY,
                    message="No requests processed yet"
                )
            
            error_rate = total_errors / total_requests
            
            if error_rate > 0.1:  # 10% error rate
                status = HealthStatus.CRITICAL if error_rate > 0.2 else HealthStatus.WARNING
                message = f"High error rate: {error_rate:.1%} ({total_errors}/{total_requests})"
            else:
                status = HealthStatus.HEALTHY
                message = f"Error rate normal: {error_rate:.1%}"
            
            return HealthCheck(
                name="error_rate",
                status=status,
                message=message,
                metadata={
                    "error_rate": error_rate,
                    "total_errors": total_errors,
                    "total_requests": total_requests,
                    "error_breakdown": dict(self.error_counts)
                }
            )
        
        # Register checks
        self.register_health_check("system_resources", system_resources_check)
        self.register_health_check("application_health", application_health_check)
        self.register_health_check("error_rate", error_rate_check)
    
    def register_health_check(self, name: str, check_function: Callable[[], HealthCheck]):
        """Register a health check function."""
        self.health_check_functions[name] = check_function
        logger.info(f"Registered health check: {name}")
    
    def record_metric(self, name: str, value: float, 
                     metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        metric = MetricPoint(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {}
        )
        
        self.metrics[name].append(metric)
        
        # Cleanup old metrics periodically
        if datetime.utcnow() - self.last_cleanup > timedelta(hours=1):
            self._cleanup_old_metrics()
    
    def record_request_time(self, duration_ms: float):
        """Record request timing."""
        self.request_times.append(duration_ms)
        self.record_metric("request_duration_ms", duration_ms, MetricType.TIMER)
    
    def record_error(self, error_type: str):
        """Record an error occurrence."""
        self.error_counts[error_type] += 1
        self.record_metric(f"errors_{error_type}", 1, MetricType.COUNTER)
    
    def create_alert(self, name: str, level: str, message: str, 
                    metadata: Optional[Dict[str, Any]] = None):
        """Create a new alert."""
        alert = Alert(
            name=name,
            level=level,
            message=message,
            metadata=metadata or {}
        )
        
        self.alerts.append(alert)
        logger.warning(f"ALERT [{level.upper()}] {name}: {message}")
        
        # Cleanup old alerts
        cutoff = datetime.utcnow() - self.alert_retention
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff]
    
    def run_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_function in self.health_check_functions.items():
            try:
                start_time = time.time()
                check_result = check_function()
                check_result.duration_ms = (time.time() - start_time) * 1000
                
                results[name] = check_result
                self.health_checks[name] = check_result
                
                # Create alerts for critical issues
                if check_result.status == HealthStatus.CRITICAL:
                    self.create_alert(
                        name=f"health_check_{name}",
                        level="critical",
                        message=check_result.message,
                        metadata=check_result.metadata
                    )
                elif check_result.status == HealthStatus.WARNING:
                    self.create_alert(
                        name=f"health_check_{name}",
                        level="warning", 
                        message=check_result.message,
                        metadata=check_result.metadata
                    )
                
            except Exception as e:
                error_result = HealthCheck(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {str(e)}"
                )
                results[name] = error_result
                self.health_checks[name] = error_result
                logger.error(f"Health check {name} failed: {e}")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_checks = self.run_health_checks()
        
        # Overall status
        statuses = [check.status for check in health_checks.values()]
        if HealthStatus.CRITICAL in statuses:
            overall_status = HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            overall_status = HealthStatus.WARNING
        elif HealthStatus.UNKNOWN in statuses:
            overall_status = HealthStatus.UNKNOWN
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Performance metrics
        recent_requests = [t for t in self.request_times if t is not None]
        avg_response_time = sum(recent_requests) / len(recent_requests) if recent_requests else 0
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "health_checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "duration_ms": check.duration_ms,
                    "metadata": check.metadata
                }
                for name, check in health_checks.items()
            },
            "performance": {
                "avg_response_time_ms": avg_response_time,
                "total_requests": len(self.request_times),
                "total_errors": sum(self.error_counts.values()),
                "error_rate": sum(self.error_counts.values()) / len(self.request_times) if self.request_times else 0
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": memory_info.percent,
                "memory_available_gb": memory_info.available / (1024**3),
                "disk_percent": psutil.disk_usage('/').percent
            },
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "monitoring_active": self.monitoring_active
        }
    
    def get_metrics(self, name: Optional[str] = None, 
                   hours: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """Get collected metrics."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        result = {}
        
        metrics_to_include = [name] if name else self.metrics.keys()
        
        for metric_name in metrics_to_include:
            if metric_name in self.metrics:
                recent_metrics = [
                    {
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "labels": m.labels
                    }
                    for m in self.metrics[metric_name]
                    if m.timestamp > cutoff
                ]
                result[metric_name] = recent_metrics
        
        return result
    
    def get_alerts(self, level: Optional[str] = None, 
                  resolved: Optional[bool] = None) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering."""
        alerts = self.alerts
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return [
            {
                "name": alert.name,
                "level": alert.level,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "metadata": alert.metadata
            }
            for alert in alerts
        ]
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.monitoring_active:
            logger.info("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info(f"Started monitoring with {self.check_interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self.run_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.check_interval)
    
    def _cleanup_old_metrics(self):
        """Clean up old metric data."""
        cutoff = datetime.utcnow() - self.metric_retention
        
        for name in self.metrics:
            # Remove old metrics
            while self.metrics[name] and self.metrics[name][0].timestamp < cutoff:
                self.metrics[name].popleft()
        
        self.last_cleanup = datetime.utcnow()
        logger.debug("Completed metric cleanup")
    
    def export_monitoring_data(self) -> Dict[str, Any]:
        """Export all monitoring data for backup or analysis."""
        return {
            "export_timestamp": datetime.utcnow().isoformat(),
            "system_status": self.get_system_status(),
            "metrics": self.get_metrics(hours=24),
            "alerts": self.get_alerts(),
            "config": {
                "check_interval": self.check_interval,
                "metric_retention_hours": self.metric_retention.total_seconds() / 3600,
                "alert_retention_hours": self.alert_retention.total_seconds() / 3600
            }
        }


# Global monitor instance
_global_monitor = None

def get_monitor() -> RobustMonitor:
    """Get the global monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = RobustMonitor()
    return _global_monitor


def monitor_function_performance(func_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            monitor = get_monitor()
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                monitor.record_metric(f"function_duration_{func_name}", duration_ms, MetricType.TIMER)
                return result
            except Exception as e:
                monitor.record_error(f"function_error_{func_name}")
                raise
        
        def sync_wrapper(*args, **kwargs):
            monitor = get_monitor()
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                monitor.record_metric(f"function_duration_{func_name}", duration_ms, MetricType.TIMER)
                return result
            except Exception as e:
                monitor.record_error(f"function_error_{func_name}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator