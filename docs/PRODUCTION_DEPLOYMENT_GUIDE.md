# Production Deployment Guide

**Agent Skeptic Bench - Autonomous SDLC v4.0**

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/agent-skeptic-bench.git
cd agent-skeptic-bench

# Deploy to production
./deployment/production-deploy.sh
```

## 📋 Prerequisites

### System Requirements
- **Kubernetes**: v1.24+ with RBAC enabled
- **Docker**: v20.10+ with BuildKit support
- **Helm**: v3.8+ for package management
- **Storage**: 500GB+ persistent storage
- **Network**: Load balancer with SSL termination

### Tool Dependencies
- `kubectl` - Kubernetes CLI
- `helm` - Package manager
- `docker` - Container runtime
- `jq` - JSON processor
- `curl` - HTTP client

### Cloud Provider Setup
Configure your cloud provider credentials:

```bash
# AWS
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# GCP
gcloud auth login
gcloud config set project your-project-id

# Azure
az login
az account set --subscription your-subscription-id
```

## 🏗️ Deployment Architecture

### Infrastructure Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (NGINX)                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │   API Pod   │  │   API Pod   │  │   API Pod   │       │
│  │             │  │             │  │             │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐                         │
│  │ Quantum     │  │ Research    │                         │
│  │ Worker      │  │ Worker      │                         │
│  └─────────────┘  └─────────────┘                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │ PostgreSQL  │  │    Redis    │  │ Monitoring  │       │
│  │ (Primary)   │  │  (Cache)    │  │ (Prometheus)│       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Service Architecture

- **API Gateway**: NGINX with SSL termination and rate limiting
- **Application Layer**: 3-20 auto-scaling replicas
- **Worker Layer**: Specialized quantum and research workers
- **Data Layer**: PostgreSQL with read replicas
- **Cache Layer**: Redis cluster with intelligent eviction
- **Monitoring**: Prometheus, Grafana, Jaeger stack

## 🔧 Configuration

### Environment Variables

Create a `.env` file with your configuration:

```bash
# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
VERSION=latest

# Database
DB_PASSWORD=your-secure-password
DATABASE_URL=postgresql://skeptic:${DB_PASSWORD}@postgres:5432/skeptic_bench

# Redis
REDIS_URL=redis://redis:6379

# Monitoring
PROMETHEUS_ENDPOINT=http://prometheus:9090
GRAFANA_PASSWORD=your-grafana-password

# Global Deployment
GLOBAL_DEPLOYMENT=true
COMPLIANCE_FRAMEWORKS=gdpr,ccpa,pdpa
SUPPORTED_LANGUAGES=en,es,fr,de,ja,zh

# Security
JWT_SECRET=your-jwt-secret-key
```

### Kubernetes Configuration

Update the deployment configuration:

```yaml
# deployment/kubernetes-production.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: agent-skeptic-bench
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  # Add your custom configuration here
```

## 🚀 Deployment Steps

### Step 1: Prepare Environment

```bash
# Set deployment variables
export DEPLOYMENT_ENV=production
export VERSION=v1.0.0
export NAMESPACE=agent-skeptic-bench
export CLUSTER_NAME=production-cluster

# Verify cluster access
kubectl cluster-info
kubectl get nodes
```

### Step 2: Build and Push Images

```bash
# Build all images
./deployment/production-deploy.sh

# Or build individually
docker build -t your-registry/agent-skeptic-bench:latest .
docker push your-registry/agent-skeptic-bench:latest
```

### Step 3: Deploy Infrastructure

```bash
# Create namespace
kubectl create namespace agent-skeptic-bench

# Deploy secrets
kubectl create secret generic app-secrets \
  --namespace=agent-skeptic-bench \
  --from-literal=DB_PASSWORD="$(openssl rand -base64 32)" \
  --from-literal=GRAFANA_PASSWORD="$(openssl rand -base64 16)" \
  --from-literal=JWT_SECRET="$(openssl rand -base64 32)"

# Deploy application
kubectl apply -f deployment/kubernetes-production.yaml
```

### Step 4: Setup Monitoring

```bash
# Add Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install monitoring stack
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --wait
```

### Step 5: Configure Ingress

```bash
# Install nginx-ingress
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Wait for deployments
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=120s
```

### Step 6: Verify Deployment

```bash
# Check deployment status
kubectl get pods -n agent-skeptic-bench
kubectl get services -n agent-skeptic-bench
kubectl get ingress -n agent-skeptic-bench

# Test health endpoints
kubectl run test-pod --rm -i --restart=Never --image=curlimages/curl -- \
  curl -f http://agent-skeptic-bench-service/health

# View logs
kubectl logs -f deployment/agent-skeptic-bench -n agent-skeptic-bench
```

## 📊 Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
- job_name: 'agent-skeptic-bench'
  static_configs:
  - targets: ['agent-skeptic-bench-service:8001']
  metrics_path: '/metrics'
  scrape_interval: 30s
```

### Grafana Dashboards

Import the provided dashboard:

```bash
# Import dashboard
kubectl create configmap grafana-dashboard \
  --from-file=deployment/grafana-dashboard.json \
  -n monitoring
```

Key metrics to monitor:
- **Response Time**: API endpoint latency
- **Throughput**: Requests per second
- **Error Rate**: HTTP 4xx/5xx responses
- **Resource Usage**: CPU, memory, disk
- **Queue Length**: Background job processing
- **Database Performance**: Query latency, connections

### Alerting Rules

```yaml
# monitoring/alert-rules.yml
groups:
- name: agent-skeptic-bench
  rules:
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
  
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
```

## 🔒 Security Configuration

### TLS/SSL Setup

```bash
# Create TLS certificate
kubectl create secret tls agent-skeptic-bench-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  -n agent-skeptic-bench
```

### Network Policies

```yaml
# security/network-policy.yml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agent-skeptic-bench-netpol
  namespace: agent-skeptic-bench
spec:
  podSelector:
    matchLabels:
      app: agent-skeptic-bench
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
```

### RBAC Configuration

```yaml
# security/rbac.yml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: agent-skeptic-bench
  name: agent-skeptic-bench-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch"]
```

## ⚡ Auto-Scaling Configuration

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agent-skeptic-bench-hpa
  namespace: agent-skeptic-bench
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-skeptic-bench
  minReplicas: 3
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

### Vertical Pod Autoscaler

```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: agent-skeptic-bench-vpa
  namespace: agent-skeptic-bench
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-skeptic-bench
  updatePolicy:
    updateMode: "Auto"
```

## 🗄️ Database Management

### PostgreSQL Setup

```bash
# Create database initialization script
kubectl create configmap postgres-init \
  --from-file=deployment/postgres/init.sql \
  -n agent-skeptic-bench

# Deploy PostgreSQL with persistence
kubectl apply -f deployment/postgres-statefulset.yaml
```

### Backup Configuration

```bash
# Setup automated backups
kubectl create cronjob postgres-backup \
  --image=postgres:15-alpine \
  --schedule="0 2 * * *" \
  --restart=OnFailure \
  -- /bin/bash -c 'pg_dump $DATABASE_URL | gzip > /backup/postgres-$(date +%Y%m%d).sql.gz'
```

### Read Replicas

```yaml
# database/read-replica.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres-replica
  namespace: agent-skeptic-bench
spec:
  replicas: 2
  selector:
    matchLabels:
      app: postgres-replica
  template:
    metadata:
      labels:
        app: postgres-replica
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_MASTER_SERVICE
          value: postgres-service
        - name: POSTGRES_REPLICATION_MODE
          value: slave
```

## 🌍 Global Deployment

### Multi-Region Setup

Deploy to multiple regions for global availability:

```bash
# Deploy to US East
REGION=us-east-1 ./deployment/production-deploy.sh

# Deploy to EU West
REGION=eu-west-1 ./deployment/production-deploy.sh

# Deploy to Asia Pacific
REGION=ap-southeast-1 ./deployment/production-deploy.sh
```

### Traffic Routing

Configure global load balancing:

```yaml
# global/traffic-policy.yml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: agent-skeptic-bench-destination
spec:
  host: agent-skeptic-bench-service
  trafficPolicy:
    loadBalancer:
      simple: LEAST_CONN
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        maxRequestsPerConnection: 2
```

### Data Residency

Configure region-specific data handling:

```yaml
# global/data-residency.yml
apiVersion: v1
kind: ConfigMap
metadata:
  name: regional-config
  namespace: agent-skeptic-bench
data:
  US_EAST_CONFIG: |
    compliance_frameworks: ["ccpa"]
    data_residency: true
    encryption_required: true
  EU_WEST_CONFIG: |
    compliance_frameworks: ["gdpr"]
    data_residency: true
    encryption_required: true
    right_to_deletion: true
```

## 🔧 Troubleshooting

### Common Issues

#### Pod Startup Failures
```bash
# Check pod status
kubectl get pods -n agent-skeptic-bench
kubectl describe pod <pod-name> -n agent-skeptic-bench

# Check logs
kubectl logs <pod-name> -n agent-skeptic-bench --previous
```

#### Database Connection Issues
```bash
# Test database connectivity
kubectl exec -it deployment/postgres -n agent-skeptic-bench -- psql -U skeptic -d skeptic_bench -c "SELECT 1;"

# Check database logs
kubectl logs deployment/postgres -n agent-skeptic-bench
```

#### Performance Issues
```bash
# Check resource usage
kubectl top pods -n agent-skeptic-bench
kubectl top nodes

# Check HPA status
kubectl get hpa -n agent-skeptic-bench
kubectl describe hpa agent-skeptic-bench-hpa -n agent-skeptic-bench
```

### Diagnostic Commands

```bash
# Get cluster information
kubectl cluster-info
kubectl get nodes -o wide

# Check deployment health
kubectl get deployments -n agent-skeptic-bench -o wide
kubectl get pods -n agent-skeptic-bench -o wide

# Check services and ingress
kubectl get services -n agent-skeptic-bench
kubectl get ingress -n agent-skeptic-bench

# Check persistent volumes
kubectl get pv
kubectl get pvc -n agent-skeptic-bench

# Check events
kubectl get events -n agent-skeptic-bench --sort-by='.lastTimestamp'
```

## 📈 Performance Tuning

### Resource Optimization

```yaml
# Optimized resource requests and limits
resources:
  requests:
    cpu: 1000m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 2Gi
```

### JVM Tuning (if applicable)

```bash
# Java application tuning
export JAVA_OPTS="-Xms1g -Xmx2g -XX:+UseG1GC -XX:MaxGCPauseMillis=200"
```

### Database Optimization

```sql
-- PostgreSQL optimization
-- shared_preload_libraries = 'pg_stat_statements'
-- max_connections = 100
-- shared_buffers = 256MB
-- effective_cache_size = 1GB
-- work_mem = 4MB
-- maintenance_work_mem = 64MB
```

## 🔄 Maintenance Procedures

### Rolling Updates

```bash
# Update application
kubectl set image deployment/agent-skeptic-bench \
  agent-skeptic-bench=your-registry/agent-skeptic-bench:v1.1.0 \
  -n agent-skeptic-bench

# Monitor rollout
kubectl rollout status deployment/agent-skeptic-bench -n agent-skeptic-bench

# Rollback if needed
kubectl rollout undo deployment/agent-skeptic-bench -n agent-skeptic-bench
```

### Backup Procedures

```bash
# Database backup
kubectl exec -it deployment/postgres -n agent-skeptic-bench -- \
  pg_dump -U skeptic skeptic_bench | gzip > backup-$(date +%Y%m%d).sql.gz

# Application data backup
kubectl exec -it deployment/agent-skeptic-bench -n agent-skeptic-bench -- \
  tar czf /tmp/app-data-$(date +%Y%m%d).tar.gz /app/data
```

### Security Updates

```bash
# Update base images
docker pull postgres:15-alpine
docker pull redis:7-alpine
docker pull nginx:alpine

# Rebuild and redeploy
./deployment/production-deploy.sh
```

## 📞 Support and Contact

### Documentation
- **Architecture**: [docs/ARCHITECTURE.md](ARCHITECTURE.md)
- **API Reference**: [docs/API_REFERENCE.md](API_REFERENCE.md)
- **Security**: [docs/SECURITY.md](SECURITY.md)

### Support Channels
- **Issues**: [GitHub Issues](https://github.com/yourusername/agent-skeptic-bench/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/agent-skeptic-bench/discussions)
- **Email**: support@agent-skeptic-bench.org

### Emergency Contacts
- **On-Call**: +1-XXX-XXX-XXXX
- **Security**: security@agent-skeptic-bench.org
- **Operations**: ops@agent-skeptic-bench.org

---

**Production Deployment Guide - Agent Skeptic Bench v4.0**  
*Built with ❤️ and ⚛️ by the Terragon Labs Team*