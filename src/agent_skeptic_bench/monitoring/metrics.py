"""Metrics collection and monitoring for Agent Skeptic Bench."""

import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any

# Handle optional dependencies with graceful fallbacks
try:
    from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server
    prometheus_available = True
except ImportError:
    # Fallback stubs for prometheus_client
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def inc(self, value=1): pass

    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def observe(self, value): pass

    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def labels(self, **kwargs): return self
        def set(self, value): pass
        def inc(self, value=1): pass

    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, data): pass

    def start_http_server(port): pass

    prometheus_available = False

try:
    import psutil
    psutil_available = True
except ImportError:
    # Fallback stubs for psutil
    class _MockMemory:
        def __init__(self):
            self.percent = 0.0
            self.used = 0

    class _MockDisk:
        def __init__(self):
            self.used = 0
            self.total = 1

    class _MockNetworkIO:
        def __init__(self):
            self.bytes_sent = 0
            self.bytes_recv = 0
            self.packets_sent = 0
            self.packets_recv = 0

    class _MockProcess:
        def memory_info(self):
            class MemInfo:
                rss = 0
                vms = 0
            return MemInfo()
        def num_threads(self): return 1
        def open_files(self): return []

    class _MockPsutil:
        def cpu_percent(self, interval=None): return 0.0
        def virtual_memory(self): return _MockMemory()
        def disk_usage(self, path): return _MockDisk()
        def net_io_counters(self): return _MockNetworkIO()
        def pids(self): return [1]
        def Process(self): return _MockProcess()

    psutil = _MockPsutil()
    psutil_available = False


logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Represents a metric value with metadata."""

    name: str
    value: float
    labels: dict[str, str]
    timestamp: float
    help_text: str = ""


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: dict[str, Any] = {}
        self._start_time = time.time()

    def increment_counter(self, name: str, value: float = 1, labels: dict[str, str] | None = None) -> None:
        """Increment a counter metric."""
        if name not in self.metrics:
            self.metrics[name] = {"type": "counter", "value": 0, "labels": labels or {}}

        self.metrics[name]["value"] += value
        self.metrics[name]["timestamp"] = time.time()

    def set_gauge(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        """Set a gauge metric."""
        self.metrics[name] = {
            "type": "gauge",
            "value": value,
            "labels": labels or {},
            "timestamp": time.time()
        }

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
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

    def get_metric(self, name: str) -> dict[str, Any] | None:
        """Get a metric by name."""
        return self.metrics.get(name)

    def get_all_metrics(self) -> dict[str, Any]:
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
        if not psutil_available:
            logger.debug("System metrics not available (psutil not installed)")
            return

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
        if not prometheus_available:
            logger.warning("Prometheus metrics server not available (prometheus_client not installed)")
            return

        try:
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")


def metrics_timer(metric_name: str, labels: dict[str, str] | None = None):
    """Decorator to time function execution and record metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                status = "success"
                return result
            except Exception:
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
            except Exception:
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
_metrics_collector: MetricsCollector | None = None
_prometheus_metrics: PrometheusMetrics | None = None


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
