# Monitoring and Observability

Comprehensive monitoring setup for Agent Skeptic Bench in production environments.

## Metrics Collection

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'agent-skeptic-bench'
    static_configs:
      - targets: ['skeptic-bench:9090']
    scrape_interval: 10s
    metrics_path: /metrics
    
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### Custom Metrics

```python
# src/agent_skeptic_bench/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import logging

# Define metrics
evaluations_total = Counter(
    'skeptic_bench_evaluations_total',
    'Total number of evaluations performed',
    ['model', 'category', 'status']
)

evaluation_duration = Histogram(
    'skeptic_bench_evaluation_duration_seconds',
    'Time spent on evaluations',
    ['model', 'category']
)

active_evaluations = Gauge(
    'skeptic_bench_active_evaluations',
    'Number of currently active evaluations'
)

api_requests_total = Counter(
    'skeptic_bench_api_requests_total',
    'Total API requests made',
    ['provider', 'model', 'status']
)

skepticism_scores = Histogram(
    'skeptic_bench_skepticism_scores',
    'Distribution of skepticism scores',
    ['category', 'model'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

class MetricsMiddleware:
    """Middleware to collect metrics during evaluations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def record_evaluation(self, model: str, category: str, duration: float, 
                         status: str, skepticism_score: float = None):
        """Record evaluation metrics."""
        evaluations_total.labels(model=model, category=category, status=status).inc()
        evaluation_duration.labels(model=model, category=category).observe(duration)
        
        if skepticism_score is not None:
            skepticism_scores.labels(category=category, model=model).observe(skepticism_score)
            
        self.logger.info(
            f"Evaluation completed: model={model}, category={category}, "
            f"duration={duration:.2f}s, status={status}"
        )
    
    def record_api_request(self, provider: str, model: str, status: str):
        """Record API request metrics."""
        api_requests_total.labels(provider=provider, model=model, status=status).inc()
    
    def set_active_evaluations(self, count: int):
        """Update active evaluations gauge."""
        active_evaluations.set(count)

def start_metrics_server(port: int = 9090):
    """Start Prometheus metrics server."""
    start_http_server(port)
    logging.info(f"Metrics server started on port {port}")
```

## Logging Configuration

### Structured Logging

```python
# src/agent_skeptic_bench/logging_config.py
import logging
import json
from datetime import datetime
import sys

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.name,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'model'):
            log_entry['model'] = record.model
        if hasattr(record, 'category'):
            log_entry['category'] = record.category
        if hasattr(record, 'evaluation_id'):
            log_entry['evaluation_id'] = record.evaluation_id
            
        return json.dumps(log_entry)

def setup_logging(level: str = "INFO", enable_file_logging: bool = True):
    """Configure structured logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if enable_file_logging:
        file_handler = logging.FileHandler('/app/logs/agent-skeptic.log')
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('agent_skeptic_bench').setLevel(level.upper())
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
```

### Log Aggregation

```yaml
# fluentd-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/*skeptic-bench*.log
      pos_file /var/log/fluentd-containers.log.pos
      tag kubernetes.*
      format json
      read_from_head true
    </source>
    
    <filter kubernetes.**>
      @type kubernetes_metadata
    </filter>
    
    <match kubernetes.**>
      @type elasticsearch
      host elasticsearch.monitoring.svc.cluster.local
      port 9200
      index_name skeptic-bench
      type_name _doc
    </match>
```

## Alerting Rules

### Prometheus Alerts

```yaml
# alert_rules.yml
groups:
  - name: agent-skeptic-bench.rules
    rules:
      - alert: HighEvaluationFailureRate
        expr: |
          (
            sum(rate(skeptic_bench_evaluations_total{status="failed"}[5m])) /
            sum(rate(skeptic_bench_evaluations_total[5m]))
          ) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High evaluation failure rate detected"
          description: "{{ $value | humanizePercentage }} of evaluations are failing"
      
      - alert: LongEvaluationDuration
        expr: |
          histogram_quantile(0.95, 
            sum(rate(skeptic_bench_evaluation_duration_seconds_bucket[5m])) by (le)
          ) > 300
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Evaluations taking too long"
          description: "95th percentile evaluation time is {{ $value }}s"
      
      - alert: APIRateLimitExceeded
        expr: |
          sum(rate(skeptic_bench_api_requests_total{status="rate_limited"}[5m])) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "API rate limit exceeded"
          description: "{{ $value }} API requests per second are being rate limited"
      
      - alert: PodMemoryUsageHigh
        expr: |
          (
            container_memory_usage_bytes{pod=~".*skeptic-bench.*"} /
            container_spec_memory_limit_bytes{pod=~".*skeptic-bench.*"}
          ) > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage in skeptic-bench pod"
          description: "Pod {{ $labels.pod }} memory usage is {{ $value | humanizePercentage }}"
      
      - alert: PodCPUUsageHigh
        expr: |
          (
            rate(container_cpu_usage_seconds_total{pod=~".*skeptic-bench.*"}[5m]) /
            container_spec_cpu_quota{pod=~".*skeptic-bench.*"} * 100000
          ) > 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage in skeptic-bench pod"
          description: "Pod {{ $labels.pod }} CPU usage is {{ $value | humanizePercentage }}"
```

### Alertmanager Configuration

```yaml
# alertmanager.yml
global:
  smtp_smarthost: 'smtp.gmail.com:587'
  smtp_from: 'alerts@skeptic-bench.org'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
    - match:
        severity: warning
      receiver: 'warning-alerts'

receivers:
  - name: 'web.hook'
    webhook_configs:
      - url: 'http://slack-webhook-service:5000/hook'
  
  - name: 'critical-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts-critical'
        title: 'Critical Alert: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
    
  - name: 'warning-alerts'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts-warnings'
        title: 'Warning: {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

## Dashboards

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Agent Skeptic Bench Monitoring",
    "panels": [
      {
        "title": "Evaluation Rate",
        "type": "graph",
        "targets": [{
          "expr": "sum(rate(skeptic_bench_evaluations_total[5m])) by (model, category)",
          "legendFormat": "{{ model }} - {{ category }}"
        }]
      },
      {
        "title": "Success Rate",
        "type": "singlestat",
        "targets": [{
          "expr": "sum(rate(skeptic_bench_evaluations_total{status='success'}[5m])) / sum(rate(skeptic_bench_evaluations_total[5m]))"
        }],
        "valueName": "current",
        "format": "percentunit"
      },
      {
        "title": "Evaluation Duration",
        "type": "graph",
        "targets": [{
          "expr": "histogram_quantile(0.95, sum(rate(skeptic_bench_evaluation_duration_seconds_bucket[5m])) by (le, model))",
          "legendFormat": "95th percentile - {{ model }}"
        }]
      },
      {
        "title": "Skepticism Score Distribution",
        "type": "heatmap",
        "targets": [{
          "expr": "sum(rate(skeptic_bench_skepticism_scores_bucket[5m])) by (le, category)"
        }]
      },
      {
        "title": "API Request Status",
        "type": "graph",
        "targets": [{
          "expr": "sum(rate(skeptic_bench_api_requests_total[5m])) by (provider, status)",
          "legendFormat": "{{ provider }} - {{ status }}"
        }]
      },
      {
        "title": "Resource Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes{pod=~'.*skeptic-bench.*'}",
            "legendFormat": "Memory - {{ pod }}"
          },
          {
            "expr": "rate(container_cpu_usage_seconds_total{pod=~'.*skeptic-bench.*'}[5m])",
            "legendFormat": "CPU - {{ pod }}"
          }
        ]
      }
    ]
  }
}
```

## Distributed Tracing

### Jaeger Configuration

```python
# src/agent_skeptic_bench/tracing.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
import os

def setup_tracing(service_name: str = "agent-skeptic-bench"):
    """Configure distributed tracing."""
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Configure Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name=os.getenv("JAEGER_AGENT_HOST", "localhost"),
        agent_port=int(os.getenv("JAEGER_AGENT_PORT", "6831")),
        collector_endpoint=os.getenv("JAEGER_COLLECTOR_ENDPOINT"),
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Auto-instrument libraries
    RequestsInstrumentor().instrument()
    LoggingInstrumentor().instrument()
    
    return tracer

def trace_evaluation(func):
    """Decorator to trace evaluation functions."""
    def wrapper(*args, **kwargs):
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span(f"evaluation_{func.__name__}") as span:
            # Add attributes
            if 'model' in kwargs:
                span.set_attribute("model", kwargs['model'])
            if 'category' in kwargs:
                span.set_attribute("category", kwargs['category'])
            
            try:
                result = func(*args, **kwargs)
                span.set_attribute("success", True)
                return result
            except Exception as e:
                span.set_attribute("success", False)
                span.set_attribute("error", str(e))
                raise
    return wrapper
```

## Health Checks

### Kubernetes Health Checks

```python
# src/agent_skeptic_bench/health.py
from flask import Flask, jsonify
import logging
from datetime import datetime
from typing import Dict, Any

app = Flask(__name__)
logger = logging.getLogger(__name__)

class HealthChecker:
    """Comprehensive health checking."""
    
    def __init__(self):
        self.checks = {
            'database': self._check_database,
            'api_connectivity': self._check_api_connectivity,
            'memory': self._check_memory,
            'disk': self._check_disk,
        }
    
    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            # Add your database check here
            return {"status": "healthy", "message": "Database connection OK"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Database error: {e}"}
    
    def _check_api_connectivity(self) -> Dict[str, Any]:
        """Check external API connectivity."""
        try:
            import requests
            response = requests.get("https://api.openai.com/v1/models", timeout=5)
            if response.status_code == 200:
                return {"status": "healthy", "message": "API connectivity OK"}
            else:
                return {"status": "degraded", "message": f"API returned {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"API connectivity error: {e}"}
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return {"status": "unhealthy", "message": f"High memory usage: {memory.percent}%"}
            elif memory.percent > 80:
                return {"status": "degraded", "message": f"Memory usage: {memory.percent}%"}
            else:
                return {"status": "healthy", "message": f"Memory usage: {memory.percent}%"}
        except Exception as e:
            return {"status": "unknown", "message": f"Memory check error: {e}"}
    
    def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            percent_used = (disk.used / disk.total) * 100
            if percent_used > 90:
                return {"status": "unhealthy", "message": f"High disk usage: {percent_used:.1f}%"}
            elif percent_used > 80:
                return {"status": "degraded", "message": f"Disk usage: {percent_used:.1f}%"}
            else:
                return {"status": "healthy", "message": f"Disk usage: {percent_used:.1f}%"}
        except Exception as e:
            return {"status": "unknown", "message": f"Disk check error: {e}"}
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_status = "healthy"
        
        for check_name, check_func in self.checks.items():
            try:
                result = check_func()
                results[check_name] = result
                
                if result["status"] == "unhealthy":
                    overall_status = "unhealthy"
                elif result["status"] == "degraded" and overall_status != "unhealthy":
                    overall_status = "degraded"
                    
            except Exception as e:
                results[check_name] = {
                    "status": "unknown",
                    "message": f"Check failed: {e}"
                }
                if overall_status == "healthy":
                    overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": results
        }

health_checker = HealthChecker()

@app.route('/health')
def health():
    """Basic health check endpoint."""
    try:
        from agent_skeptic_bench import SkepticBenchmark
        SkepticBenchmark()
        return jsonify({"status": "healthy", "timestamp": datetime.utcnow().isoformat()})
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/health/detailed')
def health_detailed():
    """Detailed health check endpoint."""
    result = health_checker.run_all_checks()
    status_code = 200 if result["status"] == "healthy" else 503
    return jsonify(result), status_code

@app.route('/ready')
def readiness():
    """Readiness check for Kubernetes."""
    # Check if the service is ready to accept traffic
    try:
        # Add readiness checks here
        return jsonify({"status": "ready", "timestamp": datetime.utcnow().isoformat()})
    except Exception as e:
        return jsonify({"status": "not ready", "error": str(e)}), 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

## Performance Monitoring

### APM Integration

```python
# src/agent_skeptic_bench/apm.py
from elasticapm import Client
from elasticapm.contrib.flask import ElasticAPM
import os

def setup_apm(app):
    """Setup Elastic APM monitoring."""
    apm_config = {
        'SERVICE_NAME': 'agent-skeptic-bench',
        'SECRET_TOKEN': os.getenv('ELASTIC_APM_SECRET_TOKEN'),
        'SERVER_URL': os.getenv('ELASTIC_APM_SERVER_URL'),
        'ENVIRONMENT': os.getenv('ENVIRONMENT', 'production'),
        'DEBUG': os.getenv('ELASTIC_APM_DEBUG', 'false').lower() == 'true',
    }
    
    apm = ElasticAPM(app, **apm_config)
    return apm

# Custom performance tracking
class PerformanceTracker:
    def __init__(self):
        self.client = Client(
            service_name='agent-skeptic-bench',
            secret_token=os.getenv('ELASTIC_APM_SECRET_TOKEN'),
            server_url=os.getenv('ELASTIC_APM_SERVER_URL'),
        )
    
    def track_evaluation(self, model: str, category: str):
        """Track evaluation performance."""
        return self.client.capture_transaction(
            transaction_type='evaluation',
            custom={
                'model': model,
                'category': category
            }
        )
```

## Monitoring Stack Deployment

### Kubernetes Monitoring Stack

```yaml
# monitoring-stack.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: monitoring
---
# Prometheus deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config-volume
          mountPath: /etc/prometheus
        - name: storage-volume
          mountPath: /prometheus
      volumes:
      - name: config-volume
        configMap:
          name: prometheus-config
      - name: storage-volume
        persistentVolumeClaim:
          claimName: prometheus-storage
---
# Grafana deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:latest
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          valueFrom:
            secretKeyRef:
              name: grafana-secret
              key: admin-password
        volumeMounts:
        - name: storage
          mountPath: /var/lib/grafana
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: grafana-storage
```

This monitoring setup provides comprehensive observability for Agent Skeptic Bench, enabling proactive issue detection and performance optimization.
