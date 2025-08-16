"""Comprehensive Monitoring System for Agent Skeptic Bench.

Advanced monitoring with distributed tracing, anomaly detection,
predictive analytics, and real-time alerting capabilities.
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to monitor."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Monitoring alert."""
    alert_id: str
    name: str
    severity: AlertSeverity
    timestamp: datetime
    message: str
    metric_name: str
    current_value: float
    threshold: float
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class Threshold:
    """Metric threshold configuration."""
    metric_name: str
    operator: str  # gt, lt, eq, ne
    value: float
    severity: AlertSeverity
    duration_seconds: int = 0  # How long condition must persist
    labels_filter: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Advanced metrics collection and storage."""
    
    def __init__(self, max_points_per_metric: int = 10000):
        """Initialize metrics collector."""
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.metric_types: Dict[str, MetricType] = {}
        self.max_points_per_metric = max_points_per_metric
        self.collection_start_time = datetime.utcnow()
        
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        self._record_metric(name, MetricType.COUNTER, value, labels or {})
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        self._record_metric(name, MetricType.GAUGE, value, labels or {})
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        self._record_metric(name, MetricType.HISTOGRAM, value, labels or {})
    
    def record_timing(self, name: str, duration_ms: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric."""
        self._record_metric(f"{name}_duration_ms", MetricType.HISTOGRAM, duration_ms, labels or {})
    
    def _record_metric(self, name: str, metric_type: MetricType, value: float, labels: Dict[str, str]) -> None:
        """Internal method to record a metric."""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            value=value,
            labels=labels.copy()
        )
        
        self.metrics[name].append(point)
        self.metric_types[name] = metric_type
        
        # Trim old points if needed
        if len(self.metrics[name]) > self.max_points_per_metric:
            self.metrics[name] = self.metrics[name][-self.max_points_per_metric:]
    
    def get_metric_summary(self, name: str, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        if name not in self.metrics:
            return {}
        
        cutoff_time = datetime.utcnow() - time_window
        recent_points = [
            point for point in self.metrics[name]
            if point.timestamp > cutoff_time
        ]
        
        if not recent_points:
            return {}
        
        values = [point.value for point in recent_points]
        metric_type = self.metric_types.get(name, MetricType.GAUGE)
        
        summary = {
            'metric_name': name,
            'metric_type': metric_type.value,
            'point_count': len(recent_points),
            'time_window_hours': time_window.total_seconds() / 3600,
            'first_timestamp': recent_points[0].timestamp.isoformat(),
            'last_timestamp': recent_points[-1].timestamp.isoformat(),
        }
        
        if metric_type == MetricType.COUNTER:
            summary.update({
                'total': sum(values),
                'rate_per_second': sum(values) / time_window.total_seconds(),
                'max_value': max(values),
                'latest_value': values[-1]
            })
        
        elif metric_type in [MetricType.GAUGE, MetricType.HISTOGRAM]:
            summary.update({
                'current_value': values[-1],
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'percentile_95': np.percentile(values, 95),
                'percentile_99': np.percentile(values, 99)
            })
        
        return summary
    
    def get_all_metrics(self) -> List[str]:
        """Get list of all metric names."""
        return list(self.metrics.keys())
    
    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for name, points in self.metrics.items():
            if not points:
                continue
            
            metric_type = self.metric_types.get(name, MetricType.GAUGE)
            
            # Add TYPE and HELP comments
            lines.append(f"# TYPE {name} {metric_type.value}")
            lines.append(f"# HELP {name} Agent Skeptic Bench metric")
            
            # Add metric points (only most recent for Prometheus)
            latest_point = points[-1]
            labels_str = ""
            if latest_point.labels:
                label_parts = [f'{k}="{v}"' for k, v in latest_point.labels.items()]
                labels_str = "{" + ",".join(label_parts) + "}"
            
            timestamp_ms = int(latest_point.timestamp.timestamp() * 1000)
            lines.append(f"{name}{labels_str} {latest_point.value} {timestamp_ms}")
        
        return "\n".join(lines)


class DistributedTracer:
    """Distributed tracing for request flows."""
    
    def __init__(self):
        """Initialize distributed tracer."""
        self.active_traces: Dict[str, 'TraceContext'] = {}
        self.completed_traces: List['TraceContext'] = []
        self.max_completed_traces = 1000
        
    def start_trace(self, operation_name: str, trace_id: Optional[str] = None) -> 'TraceContext':
        """Start a new trace."""
        if trace_id is None:
            trace_id = f"trace_{int(time.time() * 1000000)}_{hash(operation_name) % 10000}"
        
        trace_context = TraceContext(
            trace_id=trace_id,
            operation_name=operation_name,
            start_time=datetime.utcnow()
        )
        
        self.active_traces[trace_id] = trace_context
        return trace_context
    
    def finish_trace(self, trace_id: str) -> None:
        """Finish a trace."""
        if trace_id in self.active_traces:
            trace_context = self.active_traces[trace_id]
            trace_context.end_time = datetime.utcnow()
            trace_context.duration_ms = (
                trace_context.end_time - trace_context.start_time
            ).total_seconds() * 1000
            
            # Move to completed traces
            self.completed_traces.append(trace_context)
            del self.active_traces[trace_id]
            
            # Trim old traces
            if len(self.completed_traces) > self.max_completed_traces:
                self.completed_traces = self.completed_traces[-self.max_completed_traces:]
    
    def add_span(self, trace_id: str, span_name: str, tags: Optional[Dict[str, str]] = None) -> 'Span':
        """Add a span to an existing trace."""
        if trace_id in self.active_traces:
            span = Span(
                span_id=f"span_{int(time.time() * 1000000)}_{hash(span_name) % 10000}",
                name=span_name,
                start_time=datetime.utcnow(),
                tags=tags or {}
            )
            
            self.active_traces[trace_id].spans.append(span)
            return span
        
        raise ValueError(f"Trace {trace_id} not found")
    
    def get_trace_analysis(self, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Analyze trace performance."""
        cutoff_time = datetime.utcnow() - time_window
        recent_traces = [
            trace for trace in self.completed_traces
            if trace.end_time and trace.end_time > cutoff_time
        ]
        
        if not recent_traces:
            return {}
        
        # Calculate performance statistics
        durations = [trace.duration_ms for trace in recent_traces if trace.duration_ms]
        operation_stats = defaultdict(list)
        
        for trace in recent_traces:
            if trace.duration_ms:
                operation_stats[trace.operation_name].append(trace.duration_ms)
        
        analysis = {
            'total_traces': len(recent_traces),
            'average_duration_ms': statistics.mean(durations) if durations else 0,
            'median_duration_ms': statistics.median(durations) if durations else 0,
            'p95_duration_ms': np.percentile(durations, 95) if durations else 0,
            'p99_duration_ms': np.percentile(durations, 99) if durations else 0,
            'slowest_traces': self._get_slowest_traces(recent_traces, 5),
            'operation_performance': {}
        }
        
        # Per-operation statistics
        for operation, op_durations in operation_stats.items():
            analysis['operation_performance'][operation] = {
                'count': len(op_durations),
                'avg_duration_ms': statistics.mean(op_durations),
                'p95_duration_ms': np.percentile(op_durations, 95),
                'error_rate': self._calculate_error_rate(recent_traces, operation)
            }
        
        return analysis
    
    def _get_slowest_traces(self, traces: List['TraceContext'], count: int) -> List[Dict[str, Any]]:
        """Get the slowest traces."""
        traces_with_duration = [t for t in traces if t.duration_ms]
        slowest = sorted(traces_with_duration, key=lambda t: t.duration_ms, reverse=True)[:count]
        
        return [
            {
                'trace_id': trace.trace_id,
                'operation_name': trace.operation_name,
                'duration_ms': trace.duration_ms,
                'span_count': len(trace.spans),
                'start_time': trace.start_time.isoformat()
            }
            for trace in slowest
        ]
    
    def _calculate_error_rate(self, traces: List['TraceContext'], operation: str) -> float:
        """Calculate error rate for an operation."""
        operation_traces = [t for t in traces if t.operation_name == operation]
        if not operation_traces:
            return 0.0
        
        error_count = sum(1 for trace in operation_traces if trace.has_error)
        return error_count / len(operation_traces)


@dataclass
class TraceContext:
    """Trace context for distributed tracing."""
    trace_id: str
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    spans: List['Span'] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    has_error: bool = False
    error_message: Optional[str] = None


@dataclass  
class Span:
    """Individual span within a trace."""
    span_id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    has_error: bool = False
    error_message: Optional[str] = None
    
    def finish(self, error: Optional[Exception] = None) -> None:
        """Finish the span."""
        self.end_time = datetime.utcnow()
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        
        if error:
            self.has_error = True
            self.error_message = str(error)


class AnomalyDetector:
    """Machine learning-based anomaly detection."""
    
    def __init__(self, contamination: float = 0.1):
        """Initialize anomaly detector."""
        self.contamination = contamination
        self.isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.feature_history: List[List[float]] = []
        self.is_trained = False
        self.anomaly_threshold = -0.5  # Isolation Forest threshold
        
    def add_features(self, features: List[float]) -> None:
        """Add feature vector for training."""
        self.feature_history.append(features.copy())
        
        # Keep only recent history for training
        if len(self.feature_history) > 10000:
            self.feature_history = self.feature_history[-10000:]
    
    def train(self) -> bool:
        """Train the anomaly detection model."""
        if len(self.feature_history) < 100:  # Need minimum training data
            return False
        
        try:
            # Prepare training data
            X = np.array(self.feature_history)
            X_scaled = self.scaler.fit_transform(X)
            
            # Train isolation forest
            self.isolation_forest.fit(X_scaled)
            self.is_trained = True
            
            logger.info(f"Anomaly detector trained with {len(self.feature_history)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train anomaly detector: {e}")
            return False
    
    def detect_anomaly(self, features: List[float]) -> Tuple[bool, float]:
        """Detect if features represent an anomaly."""
        if not self.is_trained:
            return False, 0.0
        
        try:
            # Scale features
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            # Get anomaly score
            anomaly_score = self.isolation_forest.decision_function(X_scaled)[0]
            is_anomaly = anomaly_score < self.anomaly_threshold
            
            # Convert score to 0-1 probability (lower score = higher anomaly probability)
            anomaly_probability = max(0.0, min(1.0, 1.0 - (anomaly_score + 1.0) / 2.0))
            
            return is_anomaly, anomaly_probability
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False, 0.0
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (simplified for Isolation Forest)."""
        if not self.is_trained or not self.feature_history:
            return {}
        
        # Calculate feature variance as proxy for importance
        X = np.array(self.feature_history)
        feature_vars = np.var(X, axis=0)
        
        # Normalize to sum to 1
        total_var = np.sum(feature_vars)
        if total_var > 0:
            feature_importance = feature_vars / total_var
        else:
            feature_importance = np.ones(len(feature_vars)) / len(feature_vars)
        
        return {f"feature_{i}": importance for i, importance in enumerate(feature_importance)}


class AlertManager:
    """Alert management and notification system."""
    
    def __init__(self):
        """Initialize alert manager."""
        self.thresholds: List[Threshold] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_callbacks: List[Callable[[Alert], None]] = []
        self.alert_suppression: Dict[str, datetime] = {}
        
    def add_threshold(self, threshold: Threshold) -> None:
        """Add a monitoring threshold."""
        self.thresholds.append(threshold)
    
    def add_notification_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a notification callback function."""
        self.notification_callbacks.append(callback)
    
    def check_thresholds(self, metrics: Dict[str, float], labels: Optional[Dict[str, str]] = None) -> List[Alert]:
        """Check metrics against thresholds and generate alerts."""
        new_alerts = []
        labels = labels or {}
        
        for threshold in self.thresholds:
            if threshold.metric_name not in metrics:
                continue
            
            # Check label filters
            if threshold.labels_filter:
                if not all(labels.get(k) == v for k, v in threshold.labels_filter.items()):
                    continue
            
            current_value = metrics[threshold.metric_name]
            threshold_violated = self._check_threshold_condition(
                current_value, threshold.operator, threshold.value
            )
            
            alert_key = f"{threshold.metric_name}_{threshold.operator}_{threshold.value}"
            
            if threshold_violated:
                # Check if this alert is already active
                if alert_key not in self.active_alerts:
                    # Check suppression
                    if self._is_alert_suppressed(alert_key):
                        continue
                    
                    # Create new alert
                    alert = Alert(
                        alert_id=f"alert_{int(time.time() * 1000)}_{hash(alert_key) % 10000}",
                        name=f"{threshold.metric_name} threshold violation",
                        severity=threshold.severity,
                        timestamp=datetime.utcnow(),
                        message=f"{threshold.metric_name} is {current_value}, threshold: {threshold.operator} {threshold.value}",
                        metric_name=threshold.metric_name,
                        current_value=current_value,
                        threshold=threshold.value,
                        labels=labels.copy()
                    )
                    
                    self.active_alerts[alert_key] = alert
                    new_alerts.append(alert)
                    
                    # Send notifications
                    self._send_notifications(alert)
            
            else:
                # Check if we can resolve an active alert
                if alert_key in self.active_alerts:
                    alert = self.active_alerts[alert_key]
                    alert.resolved = True
                    alert.resolution_time = datetime.utcnow()
                    
                    # Move to history
                    self.alert_history.append(alert)
                    del self.active_alerts[alert_key]
                    
                    # Send resolution notification
                    self._send_resolution_notification(alert)
        
        return new_alerts
    
    def _check_threshold_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Check if threshold condition is met."""
        if operator == "gt":
            return value > threshold
        elif operator == "lt":
            return value < threshold
        elif operator == "eq":
            return value == threshold
        elif operator == "ne":
            return value != threshold
        elif operator == "gte":
            return value >= threshold
        elif operator == "lte":
            return value <= threshold
        else:
            return False
    
    def _is_alert_suppressed(self, alert_key: str) -> bool:
        """Check if alert is currently suppressed."""
        if alert_key in self.alert_suppression:
            suppression_end = self.alert_suppression[alert_key]
            if datetime.utcnow() < suppression_end:
                return True
            else:
                del self.alert_suppression[alert_key]
        
        return False
    
    def suppress_alert(self, alert_key: str, duration: timedelta) -> None:
        """Suppress an alert for a specified duration."""
        self.alert_suppression[alert_key] = datetime.utcnow() + duration
    
    def _send_notifications(self, alert: Alert) -> None:
        """Send alert notifications."""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")
    
    def _send_resolution_notification(self, alert: Alert) -> None:
        """Send alert resolution notification."""
        resolution_message = f"RESOLVED: {alert.message}"
        
        # Create resolution pseudo-alert for notifications
        resolution_alert = Alert(
            alert_id=f"{alert.alert_id}_resolved",
            name=f"RESOLVED: {alert.name}",
            severity=AlertSeverity.INFO,
            timestamp=datetime.utcnow(),
            message=resolution_message,
            metric_name=alert.metric_name,
            current_value=alert.current_value,
            threshold=alert.threshold,
            labels=alert.labels,
            resolved=True,
            resolution_time=alert.resolution_time
        )
        
        self._send_notifications(resolution_alert)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > last_24h
        ]
        
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in recent_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            'active_alerts': len(self.active_alerts),
            'alerts_24h': len(recent_alerts),
            'severity_distribution': severity_counts,
            'mean_time_to_resolution': self._calculate_mean_resolution_time(recent_alerts),
            'most_frequent_alerts': self._get_most_frequent_alerts(recent_alerts)
        }
    
    def _calculate_mean_resolution_time(self, alerts: List[Alert]) -> float:
        """Calculate mean time to resolution in minutes."""
        resolved_alerts = [alert for alert in alerts if alert.resolved and alert.resolution_time]
        
        if not resolved_alerts:
            return 0.0
        
        resolution_times = [
            (alert.resolution_time - alert.timestamp).total_seconds() / 60
            for alert in resolved_alerts
        ]
        
        return statistics.mean(resolution_times)
    
    def _get_most_frequent_alerts(self, alerts: List[Alert], top_n: int = 5) -> List[Dict[str, Any]]:
        """Get most frequent alert types."""
        alert_counts = defaultdict(int)
        
        for alert in alerts:
            alert_counts[alert.metric_name] += 1
        
        most_frequent = sorted(alert_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [{'metric': metric, 'count': count} for metric, count in most_frequent]


class PerformanceMonitor:
    """Comprehensive performance monitoring."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics_collector = MetricsCollector()
        self.tracer = DistributedTracer()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager()
        self.monitoring_start_time = datetime.utcnow()
        
        # Setup default thresholds
        self._setup_default_thresholds()
        
        # Setup default notification handlers
        self._setup_default_notifications()
    
    def _setup_default_thresholds(self) -> None:
        """Setup default monitoring thresholds."""
        default_thresholds = [
            Threshold("response_time_ms", "gt", 1000, AlertSeverity.WARNING),
            Threshold("response_time_ms", "gt", 5000, AlertSeverity.ERROR),
            Threshold("error_rate", "gt", 0.05, AlertSeverity.WARNING),
            Threshold("error_rate", "gt", 0.1, AlertSeverity.ERROR),
            Threshold("cpu_usage_percent", "gt", 80, AlertSeverity.WARNING),
            Threshold("cpu_usage_percent", "gt", 95, AlertSeverity.CRITICAL),
            Threshold("memory_usage_percent", "gt", 85, AlertSeverity.WARNING),
            Threshold("memory_usage_percent", "gt", 95, AlertSeverity.CRITICAL),
            Threshold("active_requests", "gt", 1000, AlertSeverity.WARNING),
            Threshold("queue_size", "gt", 100, AlertSeverity.WARNING),
        ]
        
        for threshold in default_thresholds:
            self.alert_manager.add_threshold(threshold)
    
    def _setup_default_notifications(self) -> None:
        """Setup default notification handlers."""
        def log_alert(alert: Alert) -> None:
            level = {
                AlertSeverity.INFO: logging.INFO,
                AlertSeverity.WARNING: logging.WARNING,
                AlertSeverity.ERROR: logging.ERROR,
                AlertSeverity.CRITICAL: logging.CRITICAL
            }.get(alert.severity, logging.INFO)
            
            logger.log(level, f"ALERT: {alert.message} (ID: {alert.alert_id})")
        
        self.alert_manager.add_notification_callback(log_alert)
    
    async def monitor_operation(self, operation_name: str, operation_func: Callable, *args, **kwargs) -> Any:
        """Monitor an operation with comprehensive metrics."""
        # Start trace
        trace = self.tracer.start_trace(operation_name)
        
        # Start timing
        start_time = time.time()
        error_occurred = False
        result = None
        
        try:
            # Execute operation
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
            
            return result
            
        except Exception as e:
            error_occurred = True
            trace.has_error = True
            trace.error_message = str(e)
            
            # Record error metrics
            self.metrics_collector.record_counter(
                "operation_errors_total",
                labels={"operation": operation_name, "error_type": type(e).__name__}
            )
            
            raise
            
        finally:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            self.metrics_collector.record_timing(
                f"operation_{operation_name}",
                duration_ms,
                labels={"status": "error" if error_occurred else "success"}
            )
            
            self.metrics_collector.record_counter(
                "operations_total",
                labels={"operation": operation_name, "status": "error" if error_occurred else "success"}
            )
            
            # Finish trace
            self.tracer.finish_trace(trace.trace_id)
            
            # Check for anomalies
            await self._check_anomalies(operation_name, duration_ms, error_occurred)
    
    async def _check_anomalies(self, operation_name: str, duration_ms: float, error_occurred: bool) -> None:
        """Check for performance anomalies."""
        # Extract features for anomaly detection
        features = [
            duration_ms,
            1.0 if error_occurred else 0.0,
            len(self.tracer.active_traces),
            datetime.utcnow().hour,  # Time of day
            hash(operation_name) % 100 / 100.0  # Operation identifier
        ]
        
        # Add features to detector
        self.anomaly_detector.add_features(features)
        
        # Train detector periodically
        if len(self.anomaly_detector.feature_history) % 100 == 0:
            self.anomaly_detector.train()
        
        # Check for anomaly
        if self.anomaly_detector.is_trained:
            is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(features)
            
            if is_anomaly:
                # Record anomaly metric
                self.metrics_collector.record_counter(
                    "anomalies_detected_total",
                    labels={"operation": operation_name}
                )
                
                self.metrics_collector.record_gauge(
                    "anomaly_score",
                    anomaly_score,
                    labels={"operation": operation_name}
                )
    
    def record_system_metrics(self, cpu_percent: float, memory_percent: float, 
                            active_requests: int, queue_size: int) -> None:
        """Record system-level metrics."""
        self.metrics_collector.record_gauge("cpu_usage_percent", cpu_percent)
        self.metrics_collector.record_gauge("memory_usage_percent", memory_percent)
        self.metrics_collector.record_gauge("active_requests", active_requests)
        self.metrics_collector.record_gauge("queue_size", queue_size)
        
        # Check thresholds
        metrics = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory_percent,
            "active_requests": active_requests,
            "queue_size": queue_size
        }
        
        self.alert_manager.check_thresholds(metrics)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - self.monitoring_start_time).total_seconds()
        
        # Get recent metrics summaries
        metrics_summary = {}
        for metric_name in self.metrics_collector.get_all_metrics():
            summary = self.metrics_collector.get_metric_summary(metric_name)
            if summary:
                metrics_summary[metric_name] = summary
        
        # Get trace analysis
        trace_analysis = self.tracer.get_trace_analysis()
        
        # Get alert summary
        alert_summary = self.alert_manager.get_alert_summary()
        
        # Calculate overall health score
        health_score = self._calculate_health_score(metrics_summary, alert_summary)
        
        return {
            'overall_health_score': health_score,
            'uptime_seconds': uptime_seconds,
            'active_alerts': alert_summary['active_alerts'],
            'metrics_summary': metrics_summary,
            'trace_analysis': trace_analysis,
            'alert_summary': alert_summary,
            'anomaly_detection': {
                'is_trained': self.anomaly_detector.is_trained,
                'training_samples': len(self.anomaly_detector.feature_history),
                'contamination_rate': self.anomaly_detector.contamination
            }
        }
    
    def _calculate_health_score(self, metrics_summary: Dict[str, Any], alert_summary: Dict[str, Any]) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        score = 1.0
        
        # Penalize for active alerts
        active_alerts = alert_summary.get('active_alerts', 0)
        if active_alerts > 0:
            score -= min(0.5, active_alerts * 0.1)
        
        # Penalize for high error rates
        if 'operations_total' in metrics_summary:
            ops_summary = metrics_summary['operations_total']
            error_rate = self._calculate_error_rate_from_summary(ops_summary)
            score -= error_rate * 0.3
        
        # Penalize for high response times
        if 'operation_response_time_duration_ms' in metrics_summary:
            rt_summary = metrics_summary['operation_response_time_duration_ms']
            if rt_summary.get('p95_duration_ms', 0) > 1000:
                score -= 0.2
        
        return max(0.0, score)
    
    def _calculate_error_rate_from_summary(self, summary: Dict[str, Any]) -> float:
        """Calculate error rate from operation summary."""
        # This is a simplified calculation - in practice you'd need more detailed metrics
        return 0.0  # Placeholder
    
    def export_metrics(self, format_type: str = "prometheus") -> str:
        """Export metrics in specified format."""
        if format_type == "prometheus":
            return self.metrics_collector.export_prometheus_format()
        elif format_type == "json":
            return json.dumps(self.get_health_status(), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format_type}")


# Context managers for easy monitoring
class MonitoredOperation:
    """Context manager for monitoring operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        """Initialize monitored operation."""
        self.monitor = monitor
        self.operation_name = operation_name
        self.trace = None
        self.start_time = None
    
    async def __aenter__(self):
        """Enter async context."""
        self.trace = self.monitor.tracer.start_trace(self.operation_name)
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        duration_ms = (time.time() - self.start_time) * 1000
        error_occurred = exc_type is not None
        
        # Record metrics
        self.monitor.metrics_collector.record_timing(
            f"operation_{self.operation_name}",
            duration_ms,
            labels={"status": "error" if error_occurred else "success"}
        )
        
        if error_occurred:
            self.trace.has_error = True
            self.trace.error_message = str(exc_val)
            
            self.monitor.metrics_collector.record_counter(
                "operation_errors_total",
                labels={"operation": self.operation_name, "error_type": exc_type.__name__}
            )
        
        # Finish trace
        self.monitor.tracer.finish_trace(self.trace.trace_id)
        
        # Check anomalies
        await self.monitor._check_anomalies(self.operation_name, duration_ms, error_occurred)
    
    def add_span(self, span_name: str, tags: Optional[Dict[str, str]] = None) -> Span:
        """Add a span to the current trace."""
        return self.monitor.tracer.add_span(self.trace.trace_id, span_name, tags)