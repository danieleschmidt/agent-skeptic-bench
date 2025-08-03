"""Metrics collection and monitoring for Agent Skeptic Bench."""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from functools import wraps

from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server
import psutil


logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Represents a metric value with metadata."""
    
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: float
    help_text: str = ""


class MetricsCollector:
    """Collects and manages application metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Any] = {}
        self._start_time = time.time()
    
    def increment_counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric."""
        if name not in self.metrics:
            self.metrics[name] = {"type": "counter", "value": 0, "labels": labels or {}}
        
        self.metrics[name]["value"] += value
        self.metrics[name]["timestamp"] = time.time()
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric."""
        self.metrics[name] = {
            "type": "gauge",
            "value": value,
            "labels": labels or {},
            "timestamp": time.time()
        }
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value in a histogram metric."""
        if name not in self.metrics:
            self.metrics[name] = {
                "type": "histogram",
                "observations": [],
                "count": 0,
                "sum": 0,
                "labels": labels or {}
            }
        
        metric = self.metrics[name]
        metric["observations"].append(value)
        metric["count"] += 1
        metric["sum"] += value
        metric["timestamp"] = time.time()
        
        # Keep only last 1000 observations
        if len(metric["observations"]) > 1000:
            metric["observations"] = metric["observations"][-1000:]
    
    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a metric by name."""
        return self.metrics.get(name)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return self.metrics.copy()
    
    def get_uptime(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self._start_time


class PrometheusMetrics:
    """Prometheus-compatible metrics collector."""
    
    def __init__(self, port: int = 9090):
        """Initialize Prometheus metrics.
        
        Args:
            port: Port to serve metrics on
        """
        self.port = port
        self._init_metrics()
        
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Application info
        self.app_info = Info(
            'agent_skeptic_bench_info',
            'Agent Skeptic Bench application info'
        )
        self.app_info.info({
            'version': '1.0.0',
            'python_version': '3.13'
        })
        
        # Evaluation metrics
        self.evaluations_total = Counter(
            'evaluations_total',
            'Total number of evaluations performed',
            ['agent_provider', 'model', 'category', 'status']
        )
        
        self.evaluation_duration = Histogram(
            'evaluation_duration_seconds',
            'Time spent on evaluations',
            ['agent_provider', 'model', 'category'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        
        self.evaluation_score = Histogram(
            'evaluation_score',
            'Evaluation scores',
            ['agent_provider', 'model', 'category', 'metric_type'],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Session metrics
        self.sessions_total = Counter(
            'sessions_total',
            'Total number of benchmark sessions',
            ['agent_provider', 'model', 'status']
        )
        
        self.active_sessions = Gauge(
            'active_sessions',
            'Number of currently active sessions'
        )
        
        self.session_duration = Histogram(
            'session_duration_seconds',
            'Duration of benchmark sessions',
            ['agent_provider', 'model'],
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400]
        )
        
        # API metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Database metrics
        self.db_connections_active = Gauge(
            'db_connections_active',
            'Number of active database connections'
        )
        
        self.db_query_duration = Histogram(
            'db_query_duration_seconds',
            'Database query duration',
            ['operation', 'table'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        # Cache metrics
        self.cache_operations_total = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'backend', 'status']
        )
        
        self.cache_hit_rate = Gauge(
            'cache_hit_rate',
            'Cache hit rate'
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage'
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_bytes',
            'System memory usage in bytes'
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage'
        )
        
        # AI provider metrics
        self.ai_api_requests_total = Counter(
            'ai_api_requests_total',
            'Total AI API requests',
            ['provider', 'model', 'status']
        )
        
        self.ai_api_latency = Histogram(
            'ai_api_latency_seconds',
            'AI API request latency',
            ['provider', 'model'],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]
        )
        
        self.ai_tokens_used = Counter(
            'ai_tokens_used_total',
            'Total AI tokens used',
            ['provider', 'model', 'type']
        )
    
    def record_evaluation(self, agent_provider: str, model: str, category: str, 
                         duration: float, score: float, status: str = "success") -> None:
        """Record evaluation metrics."""
        labels = {
            'agent_provider': agent_provider,
            'model': model,
            'category': category
        }
        
        self.evaluations_total.labels(**labels, status=status).inc()
        self.evaluation_duration.labels(**labels).observe(duration)
        self.evaluation_score.labels(**labels, metric_type='overall').observe(score)
    
    def record_session(self, agent_provider: str, model: str, duration: float, 
                      status: str = "completed") -> None:
        """Record session metrics."""
        labels = {
            'agent_provider': agent_provider,
            'model': model
        }
        
        self.sessions_total.labels(**labels, status=status).inc()
        self.session_duration.labels(**labels).observe(duration)
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, 
                           duration: float) -> None:
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_ai_api_call(self, provider: str, model: str, latency: float, 
                          tokens_used: int, status: str = "success") -> None:
        """Record AI API call metrics."""
        self.ai_api_requests_total.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()
        
        self.ai_api_latency.labels(
            provider=provider,
            model=model
        ).observe(latency)
        
        self.ai_tokens_used.labels(
            provider=provider,
            model=model,
            type='total'
        ).inc(tokens_used)
    
    def update_system_metrics(self) -> None:
        """Update system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.system_cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.system_memory_usage.set(memory.used)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.system_disk_usage.set(disk_percent)
    
    def set_active_sessions(self, count: int) -> None:
        """Set number of active sessions."""
        self.active_sessions.set(count)
    
    def set_db_connections(self, count: int) -> None:
        """Set number of active database connections."""
        self.db_connections_active.set(count)
    
    def start_server(self) -> None:
        """Start Prometheus metrics server."""
        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")


def metrics_timer(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution and record metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                # Record metrics (would integrate with global metrics collector)
                logger.debug(f"Function {func.__name__} took {duration:.3f}s, status: {status}")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                logger.debug(f"Function {func.__name__} took {duration:.3f}s, status: {status}")
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global metrics instances
_metrics_collector: Optional[MetricsCollector] = None
_prometheus_metrics: Optional[PrometheusMetrics] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    
    return _metrics_collector


def get_prometheus_metrics() -> PrometheusMetrics:
    """Get global Prometheus metrics instance."""
    global _prometheus_metrics
    
    if _prometheus_metrics is None:
        _prometheus_metrics = PrometheusMetrics()
    
    return _prometheus_metrics


def start_metrics_server(port: int = 9090) -> None:
    """Start Prometheus metrics server."""
    prometheus_metrics = get_prometheus_metrics()
    prometheus_metrics.start_server()


import asyncio