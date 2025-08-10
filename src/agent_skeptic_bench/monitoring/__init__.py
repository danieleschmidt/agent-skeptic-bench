"""Monitoring and observability for Agent Skeptic Bench."""

# Handle optional monitoring dependencies with graceful fallbacks
try:
    from .metrics import MetricsCollector, PrometheusMetrics
    metrics_available = True
except ImportError:
    # Fallback stubs for missing metrics dependencies (prometheus_client, psutil)
    class MetricsCollector:
        def __init__(self):
            self.metrics = {}
            self._start_time = 0
        def increment_counter(self, name, value=1, labels=None): pass
        def set_gauge(self, name, value, labels=None): pass
        def observe_histogram(self, name, value, labels=None): pass
        def get_metric(self, name): return None
        def get_all_metrics(self): return {}
        def get_uptime(self): return 0.0

    class PrometheusMetrics:
        def __init__(self, port=9090):
            self.port = port
        def record_evaluation(self, *args, **kwargs): pass
        def record_session(self, *args, **kwargs): pass
        def record_http_request(self, *args, **kwargs): pass
        def record_ai_api_call(self, *args, **kwargs): pass
        def update_system_metrics(self): pass
        def set_active_sessions(self, count): pass
        def set_db_connections(self, count): pass
        def start_server(self): pass

    metrics_available = False

try:
    from .performance import PerformanceMetrics, PerformanceMonitor
    performance_available = True
except ImportError:
    # Fallback stubs for missing performance dependencies (psutil)
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Any, Dict, List, Optional

    @dataclass
    class PerformanceMetrics:
        timestamp: datetime
        cpu_usage: float = 0.0
        memory_usage: float = 0.0
        memory_rss: int = 0
        memory_vms: int = 0
        thread_count: int = 0
        response_time: float = 0.0
        throughput: float = 0.0
        error_rate: float = 0.0
        active_connections: int = 0
        queue_size: int = 0
        custom_metrics: dict[str, float] = None

        def __post_init__(self):
            if self.custom_metrics is None:
                self.custom_metrics = {}

    class PerformanceMonitor:
        def __init__(self, sample_interval=1.0, history_size=1000):
            self.sample_interval = sample_interval
            self.history_size = history_size
            self.is_monitoring = False
            self.monitor_task = None
        async def start_monitoring(self): pass
        async def stop_monitoring(self): pass
        def record_request(self, response_time, error=False): pass
        def set_active_connections(self, count): pass
        def set_queue_size(self, size): pass
        def set_custom_metric(self, name, value): pass
        def get_current_metrics(self): return None
        def get_metrics_history(self, minutes=60): return []
        def get_active_alerts(self): return []
        def get_resolved_alerts(self, hours=24): return []
        def get_performance_summary(self): return {"status": "unavailable"}
        def get_performance_trends(self): return {"status": "unavailable"}

    performance_available = False

try:
    from .health import HealthChecker, HealthStatus
    health_available = True
except ImportError:
    # Fallback stubs for missing health dependencies (psutil)
    from enum import Enum

    class HealthStatus(Enum):
        HEALTHY = "healthy"
        DEGRADED = "degraded"
        UNHEALTHY = "unhealthy"
        CRITICAL = "critical"

    class HealthChecker:
        def __init__(self):
            self.health_checks = {}
            self.check_history = []
        def register_health_check(self, name, check_func, component_type): pass
        async def run_all_checks(self): return {}
        def get_component_status(self, component): return HealthStatus.HEALTHY
        def get_overall_health(self): return HealthStatus.HEALTHY
        def get_health_summary(self): return {"overall_status": "healthy", "components": {}}
        def setup_default_checks(self): pass

    health_available = False

from .alerts import AlertManager, AlertRule

__all__ = [
    "MetricsCollector",
    "PrometheusMetrics",
    "HealthChecker",
    "HealthStatus",
    "PerformanceMonitor",
    "PerformanceMetrics",
    "AlertManager",
    "AlertRule",
    "metrics_available",
    "performance_available",
    "health_available"
]
