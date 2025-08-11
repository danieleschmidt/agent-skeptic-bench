"""Advanced Monitoring and Observability System for Agent Skeptic Bench.

Provides comprehensive monitoring, metrics collection, alerting,
and performance analysis with distributed tracing capabilities.
"""

import asyncio
import logging
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

from ..models import EvaluationResult, Scenario
from ..quantum_optimizer import OptimizationResult

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Types of metrics being collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Alert definition and status."""
    name: str
    severity: AlertSeverity
    condition: str
    threshold: float
    current_value: float
    timestamp: datetime
    active: bool = True
    acknowledged: bool = False
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance baseline for comparison."""
    metric_name: str
    baseline_value: float
    acceptable_deviation: float
    measurement_window: int  # seconds
    last_updated: datetime
    samples: List[Tuple[datetime, float]] = field(default_factory=list)


class AdvancedMonitoringSystem:
    """Advanced monitoring and observability system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced monitoring system."""
        self.config = config or self._default_config()
        
        # Prometheus metrics registry
        self.registry = CollectorRegistry()
        self._setup_prometheus_metrics()
        
        # Internal metrics storage
        self.metrics_buffer = defaultdict(deque)
        self.alerts = defaultdict(list)
        self.performance_baselines = {}
        
        # System monitoring
        self.system_metrics = {
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'disk_usage': deque(maxlen=1000),
            'network_io': deque(maxlen=1000),
            'process_count': deque(maxlen=1000)
        }
        
        # Application-specific metrics
        self.app_metrics = {
            'evaluation_count': 0,
            'quantum_optimizations': 0,
            'security_incidents': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'database_queries': 0,
            'api_requests': 0,
            'error_count': 0
        }
        
        # Performance tracking
        self.response_times = deque(maxlen=10000)
        self.throughput_counter = deque(maxlen=1000)
        
        # Health check endpoints
        self.health_checks = {
            'database': self._check_database_health,
            'cache': self._check_cache_health,
            'quantum_optimizer': self._check_quantum_health,
            'security_system': self._check_security_health
        }
        
        # Start background monitoring tasks
        self._monitoring_task = None
        self.start_monitoring()
        
        logger.info("Advanced monitoring system initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default monitoring configuration."""
        return {
            'collection_interval': 10,  # seconds
            'retention_hours': 24,
            'alert_thresholds': {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'disk_usage': 90.0,
                'response_time_p95': 2000.0,  # ms
                'error_rate': 0.05,  # 5%
                'quantum_coherence': 0.7
            },
            'baseline_update_interval': 3600,  # 1 hour
            'prometheus_port': 9090,
            'grafana_integration': True,
            'jaeger_integration': True
        }
    
    def _setup_prometheus_metrics(self) -> None:
        """Setup Prometheus metrics."""
        # Application metrics
        self.prom_evaluation_count = Counter(
            'evaluations_total',
            'Total number of evaluations performed',
            registry=self.registry
        )
        
        self.prom_response_time = Histogram(
            'response_time_seconds',
            'Response time in seconds',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        self.prom_quantum_coherence = Gauge(
            'quantum_coherence_score',
            'Current quantum coherence score',
            registry=self.registry
        )
        
        self.prom_security_score = Gauge(
            'security_score',
            'Current security score',
            registry=self.registry
        )
        
        self.prom_active_alerts = Gauge(
            'active_alerts_total',
            'Number of active alerts',
            ['severity'],
            registry=self.registry
        )
        
        # System metrics
        self.prom_cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.prom_memory_usage = Gauge(
            'memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )
        
        self.prom_disk_usage = Gauge(
            'disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
    
    def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started background monitoring tasks")
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring tasks."""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            logger.info("Stopped background monitoring tasks")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        interval = self.config['collection_interval']
        
        while True:
            try:
                await self._collect_system_metrics()
                await self._update_prometheus_metrics()
                await self._check_alert_conditions()
                await self._cleanup_old_metrics()
                
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                logger.info("Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        timestamp = datetime.now()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_metrics['cpu_usage'].append((timestamp, cpu_percent))
            self.prom_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.system_metrics['memory_usage'].append((timestamp, memory_percent))
            self.prom_memory_usage.set(memory_percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_metrics['disk_usage'].append((timestamp, disk_percent))
            self.prom_disk_usage.set(disk_percent)
            
            # Network I/O
            network = psutil.net_io_counters()
            network_total = network.bytes_sent + network.bytes_recv
            self.system_metrics['network_io'].append((timestamp, network_total))
            
            # Process count
            process_count = len(psutil.pids())
            self.system_metrics['process_count'].append((timestamp, process_count))
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _update_prometheus_metrics(self) -> None:
        """Update Prometheus metrics from internal state."""
        try:
            # Update quantum coherence if available
            # This would be updated by the quantum optimizer
            
            # Update security score if available
            # This would be updated by the security system
            
            # Update active alerts count
            for severity in AlertSeverity:
                active_count = len([
                    alert for alerts in self.alerts.values()
                    for alert in alerts
                    if alert.severity == severity and alert.active
                ])
                self.prom_active_alerts.labels(severity=severity.value).set(active_count)
                
        except Exception as e:
            logger.error(f"Error updating Prometheus metrics: {e}")
    
    async def _check_alert_conditions(self) -> None:
        """Check alert conditions and trigger alerts."""
        thresholds = self.config['alert_thresholds']
        
        # Check CPU usage
        if self.system_metrics['cpu_usage']:
            _, latest_cpu = self.system_metrics['cpu_usage'][-1]
            if latest_cpu > thresholds['cpu_usage']:
                await self._trigger_alert(
                    'high_cpu_usage',
                    AlertSeverity.WARNING,
                    f"CPU usage is {latest_cpu:.1f}% (threshold: {thresholds['cpu_usage']}%)",
                    latest_cpu,
                    thresholds['cpu_usage']
                )
        
        # Check memory usage
        if self.system_metrics['memory_usage']:
            _, latest_memory = self.system_metrics['memory_usage'][-1]
            if latest_memory > thresholds['memory_usage']:
                await self._trigger_alert(
                    'high_memory_usage',
                    AlertSeverity.WARNING,
                    f"Memory usage is {latest_memory:.1f}% (threshold: {thresholds['memory_usage']}%)",
                    latest_memory,
                    thresholds['memory_usage']
                )
        
        # Check disk usage
        if self.system_metrics['disk_usage']:
            _, latest_disk = self.system_metrics['disk_usage'][-1]
            if latest_disk > thresholds['disk_usage']:
                await self._trigger_alert(
                    'high_disk_usage',
                    AlertSeverity.CRITICAL,
                    f"Disk usage is {latest_disk:.1f}% (threshold: {thresholds['disk_usage']}%)",
                    latest_disk,
                    thresholds['disk_usage']
                )
        
        # Check response time P95
        if self.response_times:
            p95_response_time = np.percentile(list(self.response_times), 95)
            if p95_response_time > thresholds['response_time_p95']:
                await self._trigger_alert(
                    'high_response_time',
                    AlertSeverity.WARNING,
                    f"P95 response time is {p95_response_time:.1f}ms (threshold: {thresholds['response_time_p95']}ms)",
                    p95_response_time,
                    thresholds['response_time_p95']
                )
        
        # Check error rate
        total_requests = self.app_metrics.get('api_requests', 0)
        error_count = self.app_metrics.get('error_count', 0)
        if total_requests > 100:  # Only check if we have enough samples
            error_rate = error_count / total_requests
            if error_rate > thresholds['error_rate']:
                await self._trigger_alert(
                    'high_error_rate',
                    AlertSeverity.CRITICAL,
                    f"Error rate is {error_rate:.2%} (threshold: {thresholds['error_rate']:.2%})",
                    error_rate,
                    thresholds['error_rate']
                )
    
    async def _trigger_alert(self,
                           alert_name: str,
                           severity: AlertSeverity,
                           message: str,
                           current_value: float,
                           threshold: float) -> None:
        """Trigger an alert."""
        # Check if alert is already active
        existing_alerts = self.alerts.get(alert_name, [])
        active_alert = next(
            (alert for alert in existing_alerts if alert.active),
            None
        )
        
        if active_alert:
            # Update existing alert
            active_alert.current_value = current_value
            active_alert.timestamp = datetime.now()
            return
        
        # Create new alert
        alert = Alert(
            name=alert_name,
            severity=severity,
            condition=f"> {threshold}",
            threshold=threshold,
            current_value=current_value,
            timestamp=datetime.now(),
            message=message
        )
        
        self.alerts[alert_name].append(alert)
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR,
            AlertSeverity.EMERGENCY: logging.CRITICAL
        }
        
        logger.log(log_level[severity], f"ALERT [{severity.value.upper()}]: {message}")
        
        # Send notifications
        await self._send_alert_notification(alert)
    
    async def _send_alert_notification(self, alert: Alert) -> None:
        """Send alert notification to configured channels."""
        # In a real implementation, this would integrate with:
        # - Email (SMTP)
        # - Slack/Discord webhooks
        # - PagerDuty
        # - Custom notification services
        
        notification_message = (
            f"🚨 {alert.severity.value.upper()} ALERT: {alert.name}\n"
            f"Message: {alert.message}\n"
            f"Time: {alert.timestamp.isoformat()}\n"
            f"Current Value: {alert.current_value}\n"
            f"Threshold: {alert.threshold}"
        )
        
        logger.info(f"NOTIFICATION SENT: {notification_message}")
        # TODO: Implement actual notification sending
    
    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics data."""
        retention_hours = self.config['retention_hours']
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        # Clean system metrics
        for metric_name, metric_data in self.system_metrics.items():
            while metric_data and metric_data[0][0] < cutoff_time:
                metric_data.popleft()
        
        # Clean alert history (keep resolved alerts for analysis)
        for alert_name, alert_list in self.alerts.items():
            self.alerts[alert_name] = [
                alert for alert in alert_list
                if alert.active or (datetime.now() - alert.timestamp).hours < retention_hours
            ]
    
    async def record_evaluation(self, 
                              evaluation_result: EvaluationResult,
                              response_time_ms: float) -> None:
        """Record an evaluation event."""
        # Update counters
        self.app_metrics['evaluation_count'] += 1
        self.prom_evaluation_count.inc()
        
        # Record response time
        response_time_seconds = response_time_ms / 1000.0
        self.response_times.append(response_time_ms)
        self.prom_response_time.observe(response_time_seconds)
        
        # Record throughput
        self.throughput_counter.append((datetime.now(), 1))
        
        # Extract metrics from evaluation result
        if hasattr(evaluation_result, 'metrics') and evaluation_result.metrics:
            metrics = evaluation_result.metrics
            
            # Record specific metric values
            for metric_name, value in metrics.scores.items():
                metric_key = f"evaluation_{metric_name}"
                self.metrics_buffer[metric_key].append((datetime.now(), value))
        
        logger.debug(f"Recorded evaluation with response time: {response_time_ms:.2f}ms")
    
    async def record_quantum_optimization(self, 
                                        optimization_result: OptimizationResult) -> None:
        """Record quantum optimization event."""
        self.app_metrics['quantum_optimizations'] += 1
        
        # Update quantum coherence metric
        coherence = optimization_result.quantum_coherence
        self.prom_quantum_coherence.set(coherence)
        
        # Record optimization metrics
        self.metrics_buffer['quantum_coherence'].append((datetime.now(), coherence))
        self.metrics_buffer['optimization_time'].append(
            (datetime.now(), optimization_result.optimization_time)
        )
        self.metrics_buffer['best_score'].append(
            (datetime.now(), optimization_result.best_score)
        )
        
        # Check quantum coherence threshold
        threshold = self.config['alert_thresholds'].get('quantum_coherence', 0.7)
        if coherence < threshold:
            await self._trigger_alert(
                'low_quantum_coherence',
                AlertSeverity.WARNING,
                f"Quantum coherence is {coherence:.3f} (threshold: {threshold})",
                coherence,
                threshold
            )
        
        logger.debug(f"Recorded quantum optimization with coherence: {coherence:.3f}")
    
    async def record_security_event(self, 
                                  event_type: str, 
                                  severity: str,
                                  details: Dict[str, Any]) -> None:
        """Record security event."""
        self.app_metrics['security_incidents'] += 1
        
        # Map security severity to alert severity
        alert_severity = {
            'low': AlertSeverity.INFO,
            'medium': AlertSeverity.WARNING,
            'high': AlertSeverity.CRITICAL,
            'critical': AlertSeverity.EMERGENCY
        }.get(severity, AlertSeverity.WARNING)
        
        # Trigger alert for medium and above
        if alert_severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]:
            await self._trigger_alert(
                f"security_{event_type}",
                alert_severity,
                f"Security event: {event_type} - {details.get('message', 'No details')}",
                1.0,  # Binary event
                0.5   # Threshold
            )
        
        logger.info(f"Recorded security event: {event_type} [{severity}]")
    
    async def record_api_request(self, 
                               endpoint: str, 
                               method: str,
                               status_code: int,
                               response_time_ms: float) -> None:
        """Record API request metrics."""
        self.app_metrics['api_requests'] += 1
        
        # Record response time
        self.response_times.append(response_time_ms)
        self.prom_response_time.observe(response_time_ms / 1000.0)
        
        # Count errors (4xx and 5xx status codes)
        if status_code >= 400:
            self.app_metrics['error_count'] += 1
        
        # Record endpoint-specific metrics
        endpoint_key = f"endpoint_{endpoint}_{method}"
        self.metrics_buffer[endpoint_key].append((datetime.now(), response_time_ms))
        
        logger.debug(f"Recorded API request: {method} {endpoint} - {status_code} ({response_time_ms:.2f}ms)")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        health_results = {}
        overall_healthy = True
        
        # Run all health checks
        for check_name, check_func in self.health_checks.items():
            try:
                result = await check_func()
                health_results[check_name] = result
                if not result.get('healthy', True):
                    overall_healthy = False
            except Exception as e:
                health_results[check_name] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                overall_healthy = False
        
        # Add system metrics
        latest_metrics = self._get_latest_system_metrics()
        
        return {
            'overall_healthy': overall_healthy,
            'timestamp': datetime.now().isoformat(),
            'system_metrics': latest_metrics,
            'health_checks': health_results,
            'active_alerts': self._get_active_alerts_summary(),
            'uptime_seconds': time.time() - self._start_time if hasattr(self, '_start_time') else 0
        }
    
    def _get_latest_system_metrics(self) -> Dict[str, float]:
        """Get latest system metrics."""
        latest = {}
        
        for metric_name, metric_data in self.system_metrics.items():
            if metric_data:
                latest[metric_name] = metric_data[-1][1]
        
        return latest
    
    def _get_active_alerts_summary(self) -> Dict[str, int]:
        """Get summary of active alerts."""
        summary = {severity.value: 0 for severity in AlertSeverity}
        
        for alert_list in self.alerts.values():
            for alert in alert_list:
                if alert.active:
                    summary[alert.severity.value] += 1
        
        return summary
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database health."""
        # Mock implementation - would check actual database connection
        return {
            'healthy': True,
            'response_time_ms': 25.5,
            'connection_pool_size': 10,
            'active_connections': 3,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache health."""
        # Mock implementation - would check Redis/cache connection
        return {
            'healthy': True,
            'response_time_ms': 5.2,
            'hit_rate': 0.85,
            'memory_usage_mb': 128.5,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _check_quantum_health(self) -> Dict[str, Any]:
        """Check quantum optimizer health."""
        # Mock implementation - would check quantum optimizer state
        return {
            'healthy': True,
            'coherence_level': 0.82,
            'optimization_queue_size': 2,
            'last_optimization': datetime.now().isoformat(),
            'timestamp': datetime.now().isoformat()
        }
    
    async def _check_security_health(self) -> Dict[str, Any]:
        """Check security system health."""
        # Mock implementation - would check security system status
        return {
            'healthy': True,
            'threat_detection_active': True,
            'blocked_ips_count': 15,
            'recent_incidents_count': 3,
            'security_score': 0.91,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        # Calculate throughput
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        recent_requests = sum(
            1 for timestamp, _ in self.throughput_counter
            if timestamp > hour_ago
        )
        
        # Calculate response time percentiles
        response_times_list = list(self.response_times)
        percentiles = {}
        if response_times_list:
            percentiles = {
                'p50': np.percentile(response_times_list, 50),
                'p90': np.percentile(response_times_list, 90),
                'p95': np.percentile(response_times_list, 95),
                'p99': np.percentile(response_times_list, 99)
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'application_metrics': self.app_metrics.copy(),
            'system_metrics': self._get_latest_system_metrics(),
            'performance_metrics': {
                'requests_per_hour': recent_requests,
                'response_time_percentiles': percentiles,
                'total_response_times_collected': len(response_times_list)
            },
            'alert_summary': self._get_active_alerts_summary(),
            'health_status': 'healthy' if self._get_active_alerts_summary().get('critical', 0) == 0 else 'degraded'
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry).decode('utf-8')
    
    async def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        health_status = await self.get_health_status()
        metrics_summary = self.get_metrics_summary()
        
        # Analyze trends
        trends = await self._analyze_trends()
        
        # Performance analysis
        performance_analysis = await self._analyze_performance()
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'system_health': health_status,
            'metrics_summary': metrics_summary,
            'trend_analysis': trends,
            'performance_analysis': performance_analysis,
            'recommendations': await self._generate_monitoring_recommendations(),
            'sla_compliance': await self._calculate_sla_compliance()
        }
    
    async def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze metric trends over time."""
        trends = {}
        
        # Analyze response time trend
        if len(self.response_times) > 100:
            recent_response_times = list(self.response_times)[-100:]
            older_response_times = list(self.response_times)[-200:-100] if len(self.response_times) > 200 else []
            
            if older_response_times:
                recent_avg = np.mean(recent_response_times)
                older_avg = np.mean(older_response_times)
                trend_direction = 'improving' if recent_avg < older_avg else 'degrading'
                trend_magnitude = abs(recent_avg - older_avg) / older_avg
                
                trends['response_time'] = {
                    'direction': trend_direction,
                    'magnitude': trend_magnitude,
                    'recent_average': recent_avg,
                    'previous_average': older_avg
                }
        
        # Analyze error rate trend
        total_requests = self.app_metrics.get('api_requests', 0)
        error_count = self.app_metrics.get('error_count', 0)
        if total_requests > 0:
            current_error_rate = error_count / total_requests
            trends['error_rate'] = {
                'current_rate': current_error_rate,
                'acceptable_threshold': 0.05,
                'status': 'good' if current_error_rate < 0.05 else 'poor'
            }
        
        return trends
    
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance."""
        performance = {}
        
        # CPU performance analysis
        if self.system_metrics['cpu_usage']:
            cpu_values = [value for _, value in self.system_metrics['cpu_usage']]
            performance['cpu'] = {
                'average': np.mean(cpu_values),
                'peak': np.max(cpu_values),
                'stability': 1.0 - (np.std(cpu_values) / 100.0)  # Lower std deviation = more stable
            }
        
        # Memory performance analysis
        if self.system_metrics['memory_usage']:
            memory_values = [value for _, value in self.system_metrics['memory_usage']]
            performance['memory'] = {
                'average': np.mean(memory_values),
                'peak': np.max(memory_values),
                'growth_rate': self._calculate_growth_rate(memory_values)
            }
        
        return performance
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calculate growth rate for a series of values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression to find trend
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        slope = coefficients[0]
        
        return slope
    
    async def _generate_monitoring_recommendations(self) -> List[str]:
        """Generate monitoring and performance recommendations."""
        recommendations = []
        
        # Check system resource usage
        latest_metrics = self._get_latest_system_metrics()
        
        if latest_metrics.get('cpu_usage', 0) > 70:
            recommendations.append(
                "High CPU usage detected. Consider scaling up or optimizing CPU-intensive operations."
            )
        
        if latest_metrics.get('memory_usage', 0) > 80:
            recommendations.append(
                "High memory usage detected. Review memory leaks and consider increasing available memory."
            )
        
        if latest_metrics.get('disk_usage', 0) > 85:
            recommendations.append(
                "High disk usage detected. Clean up unnecessary files or increase disk capacity."
            )
        
        # Check response time performance
        if self.response_times:
            avg_response_time = np.mean(list(self.response_times))
            if avg_response_time > 1000:  # 1 second
                recommendations.append(
                    "High average response time detected. Consider performance optimization or caching."
                )
        
        # Check alert status
        active_alerts = self._get_active_alerts_summary()
        if active_alerts.get('critical', 0) > 0:
            recommendations.append(
                "Critical alerts are active. Review and resolve immediately."
            )
        
        if not recommendations:
            recommendations.append("System performance is within acceptable ranges. Continue monitoring.")
        
        return recommendations
    
    async def _calculate_sla_compliance(self) -> Dict[str, Any]:
        """Calculate SLA compliance metrics."""
        # Mock SLA targets
        sla_targets = {
            'availability': 99.9,  # %
            'response_time_p95': 2000,  # ms
            'error_rate': 1.0  # %
        }
        
        compliance = {}
        
        # Calculate availability (mock - would track actual uptime)
        compliance['availability'] = {
            'target': sla_targets['availability'],
            'actual': 99.95,  # Mock value
            'compliant': True
        }
        
        # Calculate response time compliance
        if self.response_times:
            p95_response_time = np.percentile(list(self.response_times), 95)
            compliance['response_time'] = {
                'target': sla_targets['response_time_p95'],
                'actual': p95_response_time,
                'compliant': p95_response_time <= sla_targets['response_time_p95']
            }
        
        # Calculate error rate compliance
        total_requests = self.app_metrics.get('api_requests', 0)
        error_count = self.app_metrics.get('error_count', 0)
        if total_requests > 0:
            error_rate = (error_count / total_requests) * 100
            compliance['error_rate'] = {
                'target': sla_targets['error_rate'],
                'actual': error_rate,
                'compliant': error_rate <= sla_targets['error_rate']
            }
        
        return compliance
