"""Performance monitoring for Agent Skeptic Bench."""

import logging
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import statistics

# Handle optional psutil dependency with graceful fallbacks
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
class PerformanceMetrics:
    """Performance metrics data structure."""
    
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_rss: int
    memory_vms: int
    thread_count: int
    response_time: float
    throughput: float
    error_rate: float
    active_connections: int
    queue_size: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceThresholds:
    """Performance alert thresholds."""
    
    cpu_warning: float = 70.0
    cpu_critical: float = 90.0
    memory_warning: float = 80.0
    memory_critical: float = 95.0
    response_time_warning: float = 1.0
    response_time_critical: float = 5.0
    error_rate_warning: float = 0.05
    error_rate_critical: float = 0.15
    throughput_min_warning: float = 10.0
    throughput_min_critical: float = 5.0


@dataclass
class PerformanceAlert:
    """Performance alert."""
    
    metric_name: str
    threshold_type: str  # 'warning' or 'critical'
    current_value: float
    threshold_value: float
    timestamp: datetime
    message: str
    resolved: bool = False


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, sample_interval: float = 1.0, history_size: int = 1000):
        """Initialize performance monitor."""
        self.sample_interval = sample_interval
        self.history_size = history_size
        self.thresholds = PerformanceThresholds()
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.active_alerts: List[PerformanceAlert] = []
        self.resolved_alerts: deque = deque(maxlen=100)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # Performance counters
        self.request_counter = 0
        self.error_counter = 0
        self.response_times: deque = deque(maxlen=100)
        self.active_connections = 0
        self.queue_size = 0
        
        # Custom metrics
        self.custom_metrics: Dict[str, Any] = {}
        
        # Start time for uptime calculation
        self.start_time = time.time()
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.is_monitoring:
            logger.warning("Performance monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = await self._collect_metrics()
                
                # Store metrics
                with self._lock:
                    self.metrics_history.append(metrics)
                
                # Check thresholds and generate alerts
                await self._check_thresholds(metrics)
                
                # Sleep until next sample
                await asyncio.sleep(self.sample_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.sample_interval)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # System metrics
        if psutil_available:
            cpu_usage = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Process metrics
            process = psutil.Process()
            memory_info = process.memory_info()
            thread_count = process.num_threads()
        else:
            # Use fallback values when psutil is not available
            cpu_usage = 0.0
            memory_usage = 0.0
            memory_info = type('MemInfo', (), {'rss': 0, 'vms': 0})()
            thread_count = 1
        
        # Application metrics
        with self._lock:
            # Calculate average response time
            avg_response_time = (
                statistics.mean(self.response_times) 
                if self.response_times else 0.0
            )
            
            # Calculate throughput (requests per second)
            throughput = self._calculate_throughput()
            
            # Calculate error rate
            error_rate = (
                self.error_counter / max(1, self.request_counter) 
                if self.request_counter > 0 else 0.0
            )
        
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_rss=memory_info.rss,
            memory_vms=memory_info.vms,
            thread_count=thread_count,
            response_time=avg_response_time,
            throughput=throughput,
            error_rate=error_rate,
            active_connections=self.active_connections,
            queue_size=self.queue_size,
            custom_metrics=self.custom_metrics.copy()
        )
    
    def _calculate_throughput(self) -> float:
        """Calculate current throughput."""
        # Get recent metrics from the last minute
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff
        ]
        
        if len(recent_metrics) < 2:
            return 0.0
        
        # Calculate requests in the time window
        time_span = (recent_metrics[-1].timestamp - recent_metrics[0].timestamp).total_seconds()
        if time_span <= 0:
            return 0.0
        
        # Estimate requests based on response time data points
        return len(self.response_times) / time_span if time_span > 0 else 0.0
    
    async def _check_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check metrics against thresholds and generate alerts."""
        # CPU usage alerts
        await self._check_metric_threshold(
            "cpu_usage", metrics.cpu_usage,
            self.thresholds.cpu_warning, self.thresholds.cpu_critical
        )
        
        # Memory usage alerts
        await self._check_metric_threshold(
            "memory_usage", metrics.memory_usage,
            self.thresholds.memory_warning, self.thresholds.memory_critical
        )
        
        # Response time alerts
        await self._check_metric_threshold(
            "response_time", metrics.response_time,
            self.thresholds.response_time_warning, self.thresholds.response_time_critical
        )
        
        # Error rate alerts
        await self._check_metric_threshold(
            "error_rate", metrics.error_rate,
            self.thresholds.error_rate_warning, self.thresholds.error_rate_critical
        )
        
        # Throughput alerts (inverted - low values are bad)
        if metrics.throughput < self.thresholds.throughput_min_critical:
            await self._generate_alert(
                "throughput", "critical", metrics.throughput,
                self.thresholds.throughput_min_critical,
                f"Throughput critically low: {metrics.throughput:.2f} req/s"
            )
        elif metrics.throughput < self.thresholds.throughput_min_warning:
            await self._generate_alert(
                "throughput", "warning", metrics.throughput,
                self.thresholds.throughput_min_warning,
                f"Throughput low: {metrics.throughput:.2f} req/s"
            )
    
    async def _check_metric_threshold(self, metric_name: str, value: float,
                                    warning_threshold: float, critical_threshold: float) -> None:
        """Check a metric against warning and critical thresholds."""
        if value >= critical_threshold:
            await self._generate_alert(
                metric_name, "critical", value, critical_threshold,
                f"{metric_name} critically high: {value:.2f}%"
            )
        elif value >= warning_threshold:
            await self._generate_alert(
                metric_name, "warning", value, warning_threshold,
                f"{metric_name} high: {value:.2f}%"
            )
        else:
            # Check if we can resolve any existing alerts for this metric
            await self._resolve_alerts(metric_name)
    
    async def _generate_alert(self, metric_name: str, threshold_type: str,
                            current_value: float, threshold_value: float, message: str) -> None:
        """Generate a performance alert."""
        # Check if similar alert already exists
        existing_alert = next(
            (alert for alert in self.active_alerts 
             if alert.metric_name == metric_name and alert.threshold_type == threshold_type),
            None
        )
        
        if existing_alert:
            # Update existing alert
            existing_alert.current_value = current_value
            existing_alert.timestamp = datetime.utcnow()
        else:
            # Create new alert
            alert = PerformanceAlert(
                metric_name=metric_name,
                threshold_type=threshold_type,
                current_value=current_value,
                threshold_value=threshold_value,
                timestamp=datetime.utcnow(),
                message=message
            )
            
            self.active_alerts.append(alert)
            logger.warning(f"Performance alert: {message}")
    
    async def _resolve_alerts(self, metric_name: str) -> None:
        """Resolve alerts for a metric that's back to normal."""
        resolved = []
        
        for alert in self.active_alerts[:]:  # Create copy to iterate safely
            if alert.metric_name == metric_name:
                alert.resolved = True
                alert.timestamp = datetime.utcnow()
                resolved.append(alert)
                self.active_alerts.remove(alert)
        
        for alert in resolved:
            self.resolved_alerts.append(alert)
            logger.info(f"Performance alert resolved: {alert.metric_name}")
    
    def record_request(self, response_time: float, error: bool = False) -> None:
        """Record a request for performance tracking."""
        with self._lock:
            self.request_counter += 1
            if error:
                self.error_counter += 1
            
            self.response_times.append(response_time)
    
    def set_active_connections(self, count: int) -> None:
        """Set the number of active connections."""
        self.active_connections = count
    
    def set_queue_size(self, size: int) -> None:
        """Set the current queue size."""
        self.queue_size = size
    
    def set_custom_metric(self, name: str, value: Any) -> None:
        """Set a custom metric value."""
        with self._lock:
            self.custom_metrics[name] = value
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent metrics."""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """Get metrics history for the specified time period."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
        with self._lock:
            return [
                m for m in self.metrics_history 
                if m.timestamp > cutoff
            ]
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get currently active alerts."""
        return self.active_alerts.copy()
    
    def get_resolved_alerts(self, hours: int = 24) -> List[PerformanceAlert]:
        """Get recently resolved alerts."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        return [
            alert for alert in self.resolved_alerts 
            if alert.timestamp > cutoff
        ]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of current performance."""
        current = self.get_current_metrics()
        
        if not current:
            return {"status": "no_data"}
        
        # Calculate statistics from recent history
        recent_metrics = self.get_metrics_history(minutes=5)
        
        if recent_metrics:
            cpu_avg = statistics.mean([m.cpu_usage for m in recent_metrics])
            memory_avg = statistics.mean([m.memory_usage for m in recent_metrics])
            response_time_avg = statistics.mean([m.response_time for m in recent_metrics])
            throughput_avg = statistics.mean([m.throughput for m in recent_metrics])
        else:
            cpu_avg = current.cpu_usage
            memory_avg = current.memory_usage
            response_time_avg = current.response_time
            throughput_avg = current.throughput
        
        return {
            "timestamp": current.timestamp.isoformat(),
            "uptime": time.time() - self.start_time,
            "current": {
                "cpu_usage": current.cpu_usage,
                "memory_usage": current.memory_usage,
                "response_time": current.response_time,
                "throughput": current.throughput,
                "error_rate": current.error_rate,
                "active_connections": current.active_connections,
                "queue_size": current.queue_size
            },
            "averages_5min": {
                "cpu_usage": cpu_avg,
                "memory_usage": memory_avg,
                "response_time": response_time_avg,
                "throughput": throughput_avg
            },
            "alerts": {
                "active_count": len(self.active_alerts),
                "critical_count": len([a for a in self.active_alerts if a.threshold_type == "critical"]),
                "warning_count": len([a for a in self.active_alerts if a.threshold_type == "warning"])
            },
            "totals": {
                "requests": self.request_counter,
                "errors": self.error_counter,
                "metrics_collected": len(self.metrics_history)
            }
        }
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends."""
        recent_metrics = self.get_metrics_history(minutes=60)
        
        if len(recent_metrics) < 10:
            return {"status": "insufficient_data"}
        
        # Calculate trends
        timestamps = [(m.timestamp - recent_metrics[0].timestamp).total_seconds() for m in recent_metrics]
        cpu_values = [m.cpu_usage for m in recent_metrics]
        memory_values = [m.memory_usage for m in recent_metrics]
        response_time_values = [m.response_time for m in recent_metrics]
        
        trends = {}
        
        # Simple linear trend calculation
        for metric_name, values in [
            ("cpu_usage", cpu_values),
            ("memory_usage", memory_values),
            ("response_time", response_time_values)
        ]:
            if len(values) > 1:
                # Calculate simple slope
                x_mean = statistics.mean(timestamps)
                y_mean = statistics.mean(values)
                
                numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(timestamps, values))
                denominator = sum((x - x_mean) ** 2 for x in timestamps)
                
                slope = numerator / denominator if denominator != 0 else 0
                
                trends[metric_name] = {
                    "slope": slope,
                    "direction": "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable",
                    "current": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "avg": statistics.mean(values)
                }
        
        return {
            "status": "ok",
            "time_period": "60 minutes",
            "trends": trends,
            "data_points": len(recent_metrics)
        }


# Performance monitoring decorator
def monitor_performance(monitor: PerformanceMonitor):
    """Decorator to monitor function performance."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            error = False
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = True
                raise
            finally:
                response_time = time.time() - start_time
                monitor.record_request(response_time, error)
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            error = False
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = True
                raise
            finally:
                response_time = time.time() - start_time
                monitor.record_request(response_time, error)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor


async def start_performance_monitoring() -> None:
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    await monitor.start_monitoring()


async def stop_performance_monitoring() -> None:
    """Stop global performance monitoring."""
    monitor = get_performance_monitor()
    await monitor.stop_monitoring()