# Operational Readiness Guide

## Overview

This guide outlines the operational requirements and procedures for running Agent Skeptic Bench in production environments. It covers deployment, monitoring, maintenance, and incident response procedures.

## Production Deployment

### Environment Requirements

#### Minimum System Requirements
- **CPU**: 4 cores, 2.4GHz or higher
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 50GB available disk space
- **Network**: Stable internet connection for AI API calls

#### Recommended Production Setup
- **CPU**: 8+ cores for concurrent evaluations
- **Memory**: 32GB RAM for large-scale benchmarks
- **Storage**: SSD with 200GB+ for results and caching
- **Network**: High-bandwidth connection (1Gbps+)

### Container Deployment

#### Docker Compose Production Setup

```yaml
version: '3.8'

services:
  agent-skeptic-bench:
    image: agent-skeptic-bench:latest
    environment:
      - NODE_ENV=production
      - LOG_LEVEL=info
      - MAX_CONCURRENT_EVALUATIONS=10
    volumes:
      - ./config:/app/config:ro
      - ./results:/app/results:rw
      - ./logs:/app/logs:rw
    restart: unless-stopped
    depends_on:
      - redis
      - postgres
      - prometheus

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: skeptic_bench
      POSTGRES_USER: skeptic
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana-dashboard.json:/etc/grafana/provisioning/dashboards/dashboard.json:ro
    restart: unless-stopped
    depends_on:
      - prometheus

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

#### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-skeptic-bench
  labels:
    app: agent-skeptic-bench
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-skeptic-bench
  template:
    metadata:
      labels:
        app: agent-skeptic-bench
    spec:
      containers:
      - name: agent-skeptic-bench
        image: agent-skeptic-bench:latest
        ports:
        - containerPort: 8000
        env:
        - name: NODE_ENV
          value: "production"
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Configuration Management

#### Environment Variables

```bash
# Production Environment Configuration
NODE_ENV=production
LOG_LEVEL=info
PORT=8000

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/skeptic_bench
REDIS_URL=redis://localhost:6379

# AI Provider Configuration
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_AI_API_KEY=...

# Rate Limiting
MAX_REQUESTS_PER_MINUTE=100
MAX_CONCURRENT_EVALUATIONS=10

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090
HEALTH_CHECK_ENABLED=true

# Security
JWT_SECRET=your-jwt-secret
CORS_ORIGINS=https://your-domain.com
TRUST_PROXY=true
```

#### Configuration Files

Create `/app/config/production.yaml`:

```yaml
server:
  port: 8000
  host: "0.0.0.0"
  cors:
    origin: ["https://your-domain.com"]
    credentials: true

database:
  host: postgres
  port: 5432
  name: skeptic_bench
  ssl: require

redis:
  host: redis
  port: 6379
  db: 0

ai_providers:
  openai:
    timeout: 30
    max_retries: 3
    rate_limit: 50
  anthropic:
    timeout: 30
    max_retries: 3
    rate_limit: 40
  google:
    timeout: 30
    max_retries: 3
    rate_limit: 30

evaluation:
  max_concurrent: 10
  timeout: 300
  cache_ttl: 3600

monitoring:
  prometheus:
    enabled: true
    port: 9090
  grafana:
    enabled: true
    port: 3000
  alerts:
    enabled: true
    webhook_url: "https://your-webhook.com"

logging:
  level: info
  format: json
  file: "/app/logs/application.log"
```

## Health Checks and Monitoring

### Health Check Endpoints

Implement the following health check endpoints:

#### `/health` - Liveness Probe
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }
```

#### `/ready` - Readiness Probe
```python
@app.get("/ready")
async def readiness_check():
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "ai_providers": await check_ai_providers()
    }
    
    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503
    
    return Response(
        content=json.dumps({
            "ready": all_ready,
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }),
        status_code=status_code,
        media_type="application/json"
    )
```

#### `/metrics` - Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
evaluation_counter = Counter('evaluations_total', 'Total evaluations', ['model', 'category'])
response_time_histogram = Histogram('response_time_seconds', 'Response time')

@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

### Monitoring and Alerting

#### Key Metrics to Monitor

**Application Metrics**:
- Request rate and response times
- Error rates by endpoint
- Evaluation success/failure rates
- AI API response times and failures
- Database connection pool usage

**System Metrics**:
- CPU utilization
- Memory usage
- Disk space utilization
- Network I/O
- Container health status

**Business Metrics**:
- Daily active evaluations
- Model performance comparisons
- Scenario completion rates
- User engagement metrics

#### Alert Rules

Create `monitoring/alert_rules.yml`:

```yaml
groups:
  - name: agent-skeptic-bench
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"

      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High response time"
          description: "95th percentile response time is {{ $value }} seconds"

      - alert: DatabaseConnectionFailure
        expr: up{job="postgres"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection failure"
          description: "Cannot connect to PostgreSQL database"

      - alert: RedisConnectionFailure
        expr: up{job="redis"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis connection failure"
          description: "Cannot connect to Redis cache"

      - alert: HighMemoryUsage
        expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Container memory usage is {{ $value | humanizePercentage }}"

      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "Disk space is {{ $value | humanizePercentage }} full"
```

## Backup and Recovery

### Database Backup

#### Automated Daily Backups
```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/app/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="skeptic_bench_backup_${TIMESTAMP}.sql"

# Create backup directory
mkdir -p $BACKUP_DIR

# Perform backup
pg_dump $DATABASE_URL > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Upload to cloud storage (optional)
aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" s3://your-backup-bucket/

# Clean up old backups (keep last 30 days)
find $BACKUP_DIR -name "*.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

#### Database Recovery
```bash
#!/bin/bash
# restore_database.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

# Stop application
docker-compose stop agent-skeptic-bench

# Restore database
gunzip -c $BACKUP_FILE | psql $DATABASE_URL

# Restart application
docker-compose start agent-skeptic-bench

echo "Database restored from: $BACKUP_FILE"
```

### Configuration Backup

```bash
#!/bin/bash
# backup_config.sh

BACKUP_DIR="/app/config-backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup configuration files
tar -czf "${BACKUP_DIR}/config_backup_${TIMESTAMP}.tar.gz" \
    config/ \
    docker-compose.yml \
    monitoring/ \
    .env

echo "Configuration backup completed"
```

## Incident Response

### Incident Classification

**P0 - Critical**: Complete service outage
**P1 - High**: Major functionality impaired
**P2 - Medium**: Minor functionality affected
**P3 - Low**: Cosmetic issues or enhancement requests

### Response Procedures

#### P0 - Critical Incidents

1. **Immediate Response** (0-5 minutes):
   - Acknowledge incident in monitoring system
   - Notify on-call team
   - Begin diagnosis

2. **Assessment** (5-15 minutes):
   - Identify root cause
   - Determine impact scope
   - Estimate recovery time

3. **Mitigation** (15-60 minutes):
   - Implement immediate fixes
   - Restore service if possible
   - Communicate status to stakeholders

4. **Resolution** (1-4 hours):
   - Deploy permanent fix
   - Verify system stability
   - Update monitoring

5. **Post-Incident** (24-48 hours):
   - Conduct post-mortem
   - Document lessons learned
   - Implement preventive measures

### Escalation Matrix

| Level | Contact | Response Time |
|-------|---------|---------------|
| L1 | On-call Engineer | 5 minutes |
| L2 | Senior Engineer | 15 minutes |
| L3 | Team Lead | 30 minutes |
| L4 | Engineering Manager | 1 hour |

### Communication Templates

#### Incident Notification
```
INCIDENT ALERT - P{{ priority }}
Service: Agent Skeptic Bench
Status: {{ status }}
Started: {{ timestamp }}
Impact: {{ description }}
ETA: {{ eta }}
```

#### Status Update
```
INCIDENT UPDATE - P{{ priority }}
Current Status: {{ status }}
Progress: {{ update }}
Next Update: {{ next_update_time }}
```

#### Resolution Notice
```
INCIDENT RESOLVED - P{{ priority }}
Resolution Time: {{ duration }}
Root Cause: {{ cause }}
Next Steps: {{ next_steps }}
```

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily
- [ ] Check system health dashboards
- [ ] Review error logs
- [ ] Verify backup completion
- [ ] Monitor resource usage

#### Weekly
- [ ] Update dependencies
- [ ] Review security alerts
- [ ] Analyze performance trends
- [ ] Update documentation

#### Monthly
- [ ] Security audit
- [ ] Capacity planning review
- [ ] Performance optimization
- [ ] Disaster recovery testing

### Deployment Procedures

#### Pre-deployment Checklist
- [ ] Code review completed
- [ ] Tests passing
- [ ] Security scan passed
- [ ] Performance impact assessed
- [ ] Rollback plan prepared
- [ ] Stakeholders notified

#### Deployment Steps
1. Create deployment branch
2. Update version numbers
3. Build and test containers
4. Deploy to staging
5. Run smoke tests
6. Deploy to production
7. Monitor for issues
8. Verify functionality

#### Post-deployment Verification
- [ ] Health checks passing
- [ ] Key metrics normal
- [ ] Error rates acceptable
- [ ] Performance within SLA
- [ ] User feedback positive

## Security Operations

### Security Monitoring

#### Daily Security Checks
- Review security scan results
- Check for failed authentication attempts
- Monitor suspicious API usage patterns
- Verify certificate status
- Review access logs

#### Security Incident Response
1. **Detection**: Automated alerts or manual discovery
2. **Containment**: Isolate affected systems
3. **Investigation**: Determine scope and impact
4. **Eradication**: Remove threats and vulnerabilities
5. **Recovery**: Restore normal operations
6. **Lessons Learned**: Update security measures

### Compliance Requirements

#### Data Protection
- Encrypt data in transit and at rest
- Implement proper access controls
- Regular security audits
- Incident reporting procedures
- Data retention policies

#### Audit Trail
- Log all authentication attempts
- Track configuration changes
- Monitor data access patterns
- Maintain deployment history
- Document security incidents

## Performance Optimization

### Performance Monitoring

#### Key Performance Indicators
- Average response time < 200ms
- 95th percentile response time < 500ms
- CPU utilization < 70%
- Memory utilization < 80%
- Database query time < 100ms

#### Performance Tuning

**Application Level**:
- Optimize database queries
- Implement caching strategies
- Use connection pooling
- Minimize external API calls
- Optimize data structures

**Infrastructure Level**:
- Scale horizontally
- Optimize container resources
- Use CDN for static assets
- Implement load balancing
- Optimize database configuration

### Capacity Planning

#### Growth Projections
- Monitor user growth trends
- Predict resource requirements
- Plan infrastructure scaling
- Budget for capacity increases
- Test scalability limits

#### Scaling Strategies
- Horizontal pod autoscaling
- Database read replicas
- Redis clustering
- CDN optimization
- API rate limiting

This operational readiness guide provides comprehensive procedures for maintaining a production-ready Agent Skeptic Bench deployment with proper monitoring, incident response, and maintenance procedures.