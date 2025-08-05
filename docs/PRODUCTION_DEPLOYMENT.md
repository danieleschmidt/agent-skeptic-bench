# Production Deployment Guide

## Overview

This guide covers deploying the Agent Skeptic Bench with quantum-inspired optimization to production environments. The system supports both Docker Compose and Kubernetes deployments with comprehensive monitoring, security, and auto-scaling capabilities.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │  Quantum Agent  │
│     (Nginx)     │────│   (FastAPI)     │────│   Evaluator     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │     Cache       │              │
         └──────────────│    (Redis)      │──────────────┘
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │    Database     │
                        │  (PostgreSQL)   │
                        └─────────────────┘
```

## Docker Compose Deployment

### Quick Start

```bash
# Clone and navigate to repository
git clone <repository-url>
cd agent-skeptic-bench

# Create production environment file
cp .env.example .env.production

# Deploy with Docker Compose
docker-compose -f deployment/docker-compose.production.yml up -d
```

### Environment Configuration

Create `.env.production` with the following variables:

```bash
# Database Configuration
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_USER=skeptic_user
POSTGRES_DB=agent_skeptic_bench

# Redis Configuration  
REDIS_PASSWORD=your_redis_password_here

# Security
JWT_SECRET_KEY=your_jwt_secret_key_here

# Monitoring
GRAFANA_PASSWORD=your_grafana_password_here

# Quantum Optimization
QUANTUM_OPTIMIZATION_ENABLED=true
AUTO_SCALING_ENABLED=true
SECURITY_VALIDATION_ENABLED=true

# Performance
CACHE_TTL=3600
RATE_LIMIT_REQUESTS_PER_MINUTE=1000
```

### Service Configuration

#### API Service
- **Image**: Built from local Dockerfile
- **Ports**: 8000 (HTTP API)
- **Resources**: 2GB RAM, 1 CPU
- **Health Checks**: `/health` endpoint
- **Auto-restart**: Always

#### Redis Cache
- **Image**: redis:7-alpine
- **Ports**: 6379
- **Persistence**: Volume-mounted data directory
- **Configuration**: Custom redis.conf for production

#### PostgreSQL Database
- **Image**: postgres:15-alpine
- **Ports**: 5432
- **Persistence**: Volume-mounted data directory
- **Initialization**: Custom SQL scripts

#### Monitoring Stack
- **Prometheus**: Metrics collection (Port 9090)
- **Grafana**: Visualization dashboard (Port 3000)
- **Jaeger**: Distributed tracing (Port 16686)

#### Load Balancer
- **Nginx**: Reverse proxy and SSL termination
- **Ports**: 80 (HTTP), 443 (HTTPS)
- **Features**: Rate limiting, compression, caching

### Scaling Configuration

The auto-scaler service monitors system metrics and adjusts container replicas:

```yaml
auto-scaler:
  environment:
    - SCALING_CHECK_INTERVAL=60
    - MIN_REPLICAS=2
    - MAX_REPLICAS=20
    - CPU_THRESHOLD_HIGH=75
    - CPU_THRESHOLD_LOW=30
    - MEMORY_THRESHOLD_HIGH=80
    - RESPONSE_TIME_THRESHOLD=2000
```

### Deployment Commands

```bash
# Start all services
docker-compose -f deployment/docker-compose.production.yml up -d

# Scale API service
docker-compose -f deployment/docker-compose.production.yml up -d --scale agent-skeptic-bench-api=3

# View logs
docker-compose -f deployment/docker-compose.production.yml logs -f agent-skeptic-bench-api

# Health check
curl http://localhost:8000/health

# Stop services
docker-compose -f deployment/docker-compose.production.yml down
```

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (v1.24+)
- kubectl configured
- Ingress controller (nginx recommended)
- Cert-manager for SSL certificates
- Prometheus operator for monitoring

### Quick Deploy

```bash
# Apply all resources
kubectl apply -f deployment/kubernetes-deployment.yaml

# Check deployment status
kubectl get pods -n agent-skeptic-bench

# Access application
kubectl port-forward -n agent-skeptic-bench svc/agent-skeptic-bench-service 8080:80
```

### Resource Breakdown

#### Namespace and Configuration
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agent-skeptic-bench
  labels:
    name: agent-skeptic-bench
    environment: production
```

#### ConfigMap
Contains environment variables for all services:
- Quantum optimization settings
- Cache configuration
- Security parameters
- Feature flags

#### Secrets
Stores sensitive data (base64 encoded):
- Database passwords
- Redis authentication
- JWT secret keys
- API keys

#### Deployments

**API Deployment**
- **Replicas**: 3 (minimum for HA)
- **Resources**: 1GB RAM request, 2GB limit
- **Probes**: Liveness and readiness checks
- **Volumes**: Data and logs persistent storage

**Redis Deployment**
- **Replicas**: 1 (single instance)
- **Resources**: 256MB RAM request, 512MB limit
- **Persistence**: 2GB storage

**PostgreSQL Deployment**
- **Replicas**: 1 (single instance)
- **Resources**: 512MB RAM request, 1GB limit
- **Persistence**: 10GB storage

#### Services
- **ClusterIP services** for internal communication
- **LoadBalancer service** for external access (optional)

#### Horizontal Pod Autoscaler (HPA)
```yaml
spec:
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Ingress Configuration
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agent-skeptic-bench-ingress
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "1000"
```

### Storage Classes

The deployment assumes these storage classes exist:
- **fast-ssd**: For database and cache (high IOPS)
- **standard**: For logs and backups

Create storage classes if needed:
```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
```

### Monitoring Integration

#### ServiceMonitor for Prometheus
```yaml
apiVersion: v1
kind: ServiceMonitor
metadata:
  name: agent-skeptic-bench-metrics
spec:
  selector:
    matchLabels:
      app: agent-skeptic-bench-api
  endpoints:
  - port: http
    path: /metrics
    interval: 30s
```

#### Network Policies
Restrict network traffic between pods:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agent-skeptic-bench-network-policy
spec:
  podSelector:
    matchLabels:
      app: agent-skeptic-bench-api
  policyTypes:
  - Ingress
  - Egress
```

## Security Configuration

### Authentication and Authorization

1. **JWT Token Authentication**
   ```python
   # Configure JWT settings
   JWT_SECRET_KEY=your-256-bit-secret
   JWT_ALGORITHM=HS256
   JWT_EXPIRATION_MINUTES=60
   ```

2. **API Key Management**
   ```bash
   # Generate API key
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

3. **Role-Based Access Control**
   ```yaml
   roles:
     - name: evaluator
       permissions: [read, evaluate]
     - name: administrator  
       permissions: [read, write, admin]
   ```

### Input Validation and Security

The system includes comprehensive security validation:

```python
# Security patterns detected
MALICIOUS_PATTERNS = [
    r'<script[^>]*>.*?</script>',  # XSS
    r'javascript:',                # JavaScript injection
    r'eval\s*\(',                 # Code evaluation
    r'exec\s*\(',                 # Code execution
    r'import\s+os',                # OS imports
    r'__import__',                 # Dynamic imports
]
```

### Network Security

1. **TLS Encryption**: All external communication uses HTTPS
2. **Network Policies**: Restrict pod-to-pod communication
3. **Firewall Rules**: Limit ingress traffic
4. **Secret Management**: Use Kubernetes secrets or external vaults

## Monitoring and Observability

### Metrics Collection

#### Application Metrics
```python
# Quantum optimization metrics
quantum_optimization_fitness = Gauge('quantum_optimization_fitness')
quantum_coherence_level = Gauge('quantum_coherence_level')
parameter_entanglement = Gauge('parameter_entanglement')

# Performance metrics
evaluation_duration = Histogram('evaluation_duration_seconds')
active_evaluations = Gauge('active_evaluations_total')
cache_hit_rate = Gauge('cache_hit_rate')
```

#### System Metrics
- CPU and memory usage
- Network I/O
- Disk usage and I/O
- Database connection pool
- Cache performance

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
- name: agent-skeptic-bench
  rules:
  - alert: HighCPUUsage
    expr: cpu_usage_percent > 80
    for: 5m
    annotations:
      summary: High CPU usage detected
      
  - alert: LowQuantumCoherence
    expr: quantum_coherence_level < 0.7
    for: 2m
    annotations:
      summary: Quantum coherence below threshold
```

### Dashboard Configuration

Grafana dashboards include:
- System performance overview
- Quantum optimization metrics
- Evaluation throughput and latency
- Error rates and success rates
- Resource utilization trends

## Performance Tuning

### Database Optimization

```sql
-- PostgreSQL configuration for production
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
```

### Cache Configuration

```redis
# Redis production configuration
maxmemory 512mb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### API Performance

```python
# FastAPI performance settings
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    # Enable gzip compression
    # Add performance headers
    # Monitor response times
```

## Backup and Recovery

### Database Backup

```bash
# Create backup
kubectl exec -n agent-skeptic-bench postgres-0 -- \
  pg_dump -U skeptic_user agent_skeptic_bench > backup.sql

# Restore backup
kubectl exec -i -n agent-skeptic-bench postgres-0 -- \
  psql -U skeptic_user agent_skeptic_bench < backup.sql
```

### Configuration Backup

```bash
# Backup Kubernetes configurations
kubectl get all,configmap,secret,pv,pvc -n agent-skeptic-bench -o yaml > k8s-backup.yaml

# Backup Docker Compose configuration
tar -czf compose-backup.tar.gz deployment/
```

### Disaster Recovery Plan

1. **Data Recovery**: Restore database from latest backup
2. **Configuration Recovery**: Redeploy from version control
3. **Service Recovery**: Rolling restart of failed components
4. **Monitoring Recovery**: Restore dashboards and alerts

## Troubleshooting

### Common Issues

#### Pod Startup Failures
```bash
# Check pod logs
kubectl logs -n agent-skeptic-bench <pod-name>

# Describe pod for events
kubectl describe pod -n agent-skeptic-bench <pod-name>

# Check resource constraints
kubectl top pods -n agent-skeptic-bench
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -n agent-skeptic-bench <api-pod> -- \
  psql -h postgres-service -U skeptic_user -d agent_skeptic_bench -c "SELECT 1;"
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n agent-skeptic-bench

# Monitor quantum optimization
curl http://api-endpoint/quantum-insights

# Check auto-scaling
kubectl get hpa -n agent-skeptic-bench
```

### Debug Mode

Enable debug logging:
```yaml
environment:
  - LOG_LEVEL=debug
  - DEBUG_QUANTUM_OPTIMIZATION=true
  - ENABLE_TRACE_LOGGING=true
```

## Maintenance

### Updates and Upgrades

```bash
# Rolling update deployment
kubectl set image deployment/agent-skeptic-bench-api \
  api=agent-skeptic-bench:v2.0.0 -n agent-skeptic-bench

# Check rollout status
kubectl rollout status deployment/agent-skeptic-bench-api -n agent-skeptic-bench

# Rollback if needed
kubectl rollout undo deployment/agent-skeptic-bench-api -n agent-skeptic-bench
```

### Health Checks

The system provides multiple health check endpoints:
- `/health`: Basic service health
- `/ready`: Readiness for traffic
- `/metrics`: Prometheus metrics
- `/quantum-health`: Quantum optimization status

### Log Management

```bash
# Centralized logging with Fluentd/ELK
kubectl logs -f -n agent-skeptic-bench -l app=agent-skeptic-bench-api

# Log rotation configuration
logrotate /var/log/agent-skeptic-bench/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
}
```

## Cost Optimization

### Resource Right-Sizing

Monitor and adjust resource requests/limits:
```yaml
resources:
  requests:
    memory: "1Gi"    # Start conservative
    cpu: "500m"      # Scale based on usage
  limits:
    memory: "2Gi"    # Prevent memory leaks
    cpu: "1000m"     # Allow burst capacity
```

### Auto-Scaling Strategy

Configure HPA for cost-effective scaling:
```yaml
behavior:
  scaleUp:
    stabilizationWindowSeconds: 60
    policies:
    - type: Percent
      value: 50
      periodSeconds: 60
  scaleDown:
    stabilizationWindowSeconds: 300
    policies:
    - type: Percent
      value: 25
      periodSeconds: 60
```

---

For detailed configuration examples, see the files in the `deployment/` directory.
For monitoring setup, refer to the Prometheus and Grafana configurations.
For security best practices, consult the `SECURITY.md` documentation.