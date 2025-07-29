# Deployment Guide

This guide covers deployment strategies for Agent Skeptic Bench in various environments.

## Docker Deployment

### Production Deployment

```bash
# Build production image
docker build -t agent-skeptic-bench:latest .

# Run container with environment variables
docker run -d \
  --name skeptic-bench \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY} \
  -v $(pwd)/results:/app/results \
  agent-skeptic-bench:latest
```

### Docker Compose

```bash
# Production deployment
docker-compose up -d

# Run benchmark evaluation
docker-compose --profile evaluation up benchmark-runner

# Development environment
docker-compose --profile dev up development
```

## Cloud Deployment

### AWS ECS

1. **Create ECR Repository**:
```bash
aws ecr create-repository --repository-name agent-skeptic-bench
```

2. **Build and Push Image**:
```bash
# Get login token
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-west-2.amazonaws.com

# Tag and push
docker tag agent-skeptic-bench:latest <account-id>.dkr.ecr.us-west-2.amazonaws.com/agent-skeptic-bench:latest
docker push <account-id>.dkr.ecr.us-west-2.amazonaws.com/agent-skeptic-bench:latest
```

3. **ECS Task Definition**:
```json
{
  "family": "agent-skeptic-bench",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "containerDefinitions": [{
    "name": "skeptic-bench",
    "image": "<account-id>.dkr.ecr.us-west-2.amazonaws.com/agent-skeptic-bench:latest",
    "essential": true,
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/agent-skeptic-bench",
        "awslogs-region": "us-west-2",
        "awslogs-stream-prefix": "ecs"
      }
    },
    "environment": [
      {"name": "PYTHONPATH", "value": "/app/src"}
    ],
    "secrets": [
      {"name": "OPENAI_API_KEY", "valueFrom": "arn:aws:secretsmanager:us-west-2:<account-id>:secret:openai-api-key"}
    ]
  }]
}
```

### Google Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy agent-skeptic-bench \
  --image gcr.io/PROJECT-ID/agent-skeptic-bench:latest \
  --platform managed \
  --region us-central1 \
  --set-env-vars PYTHONPATH=/app/src \
  --set-secrets OPENAI_API_KEY=openai-key:latest \
  --memory 2Gi \
  --cpu 2 \
  --timeout 3600
```

### Azure Container Instances

```bash
# Create resource group
az group create --name skeptic-bench-rg --location eastus

# Deploy container
az container create \
  --resource-group skeptic-bench-rg \
  --name agent-skeptic-bench \
  --image agent-skeptic-bench:latest \
  --cpu 2 \
  --memory 4 \
  --environment-variables PYTHONPATH=/app/src \
  --secure-environment-variables OPENAI_API_KEY=$OPENAI_API_KEY
```

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: skeptic-bench
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: skeptic-bench-config
  namespace: skeptic-bench
data:
  PYTHONPATH: "/app/src"
  AGENT_SKEPTIC_DEBUG: "false"
```

### Secret for API Keys

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
  namespace: skeptic-bench
type: Opaque
stringData:
  OPENAI_API_KEY: "your-openai-key"
  ANTHROPIC_API_KEY: "your-anthropic-key"
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-skeptic-bench
  namespace: skeptic-bench
spec:
  replicas: 2
  selector:
    matchLabels:
      app: agent-skeptic-bench
  template:
    metadata:
      labels:
        app: agent-skeptic-bench
    spec:
      containers:
      - name: skeptic-bench
        image: agent-skeptic-bench:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: skeptic-bench-config
        - secretRef:
            name: api-keys
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import agent_skeptic_bench; print('healthy')"
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "import agent_skeptic_bench; print('ready')"
          initialDelaySeconds: 10
          periodSeconds: 5
```

## Environment Configuration

### Production Environment Variables

```bash
# Core configuration
PYTHONPATH=/app/src
AGENT_SKEPTIC_DEBUG=false
LOG_LEVEL=INFO

# API Keys (use secrets management)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
GOOGLE_AI_API_KEY=...

# Performance tuning
MAX_CONCURRENT_EVALUATIONS=10
REQUEST_TIMEOUT=60
RETRY_ATTEMPTS=3

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
TRACING_ENABLED=true
```

### Secrets Management

#### AWS Secrets Manager
```bash
# Store API key
aws secretsmanager create-secret \
  --name "agent-skeptic-bench/openai-key" \
  --description "OpenAI API key for Agent Skeptic Bench" \
  --secret-string "sk-your-key-here"
```

#### HashiCorp Vault
```bash
# Store secrets in Vault
vault kv put secret/agent-skeptic-bench \
  openai_api_key="sk-your-key-here" \
  anthropic_api_key="your-anthropic-key"
```

## Performance Optimization

### Resource Requirements

| Deployment Size | CPU | Memory | Storage |
|----------------|-----|--------|----------|
| **Small** | 1 core | 2GB | 10GB |
| **Medium** | 2 cores | 4GB | 20GB |
| **Large** | 4 cores | 8GB | 50GB |
| **Enterprise** | 8+ cores | 16GB+ | 100GB+ |

### Scaling Configuration

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: skeptic-bench-hpa
  namespace: skeptic-bench
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agent-skeptic-bench
  minReplicas: 2
  maxReplicas: 10
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

## Monitoring and Observability

### Health Checks

```python
# Custom health check endpoint
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/health')
def health_check():
    try:
        # Test core functionality
        from agent_skeptic_bench import SkepticBenchmark
        benchmark = SkepticBenchmark()
        return jsonify({"status": "healthy", "timestamp": datetime.utcnow()})
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
```

### Logging Configuration

```python
# logging.yaml
version: 1
formatters:
  structured:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "module": "%(name)s"}'
handlers:
  console:
    class: logging.StreamHandler
    formatter: structured
    level: INFO
  file:
    class: logging.FileHandler
    filename: /app/logs/agent-skeptic.log
    formatter: structured
    level: DEBUG
loggers:
  agent_skeptic_bench:
    level: INFO
    handlers: [console, file]
    propagate: false
root:
  level: WARNING
  handlers: [console]
```

## Security Considerations

### Container Security

```dockerfile
# Security hardening
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 app

# Remove unnecessary packages
RUN apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false

# Set file permissions
COPY --chown=app:app --chmod=755 src/ /app/src/

# Run as non-root
USER app
```

### Network Security

```yaml
# Network Policy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: skeptic-bench-netpol
  namespace: skeptic-bench
spec:
  podSelector:
    matchLabels:
      app: agent-skeptic-bench
  policyTypes:
  - Ingress
  - Egress
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS only
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

## Disaster Recovery

### Backup Strategy

```bash
# Backup benchmark results
kubectl exec -n skeptic-bench deployment/agent-skeptic-bench -- \
  tar czf /tmp/results-backup-$(date +%Y%m%d).tar.gz /app/results

# Copy to persistent storage
kubectl cp skeptic-bench/pod-name:/tmp/results-backup-*.tar.gz ./backups/
```

### Recovery Procedures

1. **Container Failure**: Kubernetes automatically restarts failed containers
2. **Node Failure**: Pods are rescheduled to healthy nodes
3. **Data Loss**: Restore from latest backup in persistent volume
4. **Complete Disaster**: Redeploy from infrastructure as code

## Cost Optimization

### Resource Rightsizing

```bash
# Monitor resource usage
kubectl top pods -n skeptic-bench
kubectl describe hpa -n skeptic-bench

# Adjust resource requests/limits based on actual usage
```

### Spot Instances

```yaml
# Use spot instances for cost savings
apiVersion: v1
kind: Node
metadata:
  labels:
    node.kubernetes.io/instance-type: spot
spec:
  taints:
  - effect: NoSchedule
    key: spot-instance
    value: "true"
```

## Troubleshooting

### Common Issues

1. **Container Won't Start**:
   - Check logs: `docker logs container-name`
   - Verify environment variables
   - Check resource availability

2. **API Rate Limits**:
   - Implement exponential backoff
   - Use multiple API keys with rotation
   - Monitor API usage metrics

3. **Memory Issues**:
   - Increase memory limits
   - Optimize batch sizes
   - Check for memory leaks

### Debug Commands

```bash
# Docker debugging
docker exec -it container-name /bin/bash
docker inspect container-name
docker logs --follow container-name

# Kubernetes debugging
kubectl describe pod pod-name -n skeptic-bench
kubectl logs -f pod-name -n skeptic-bench
kubectl exec -it pod-name -n skeptic-bench -- /bin/bash
```

For additional deployment support, see our [troubleshooting guide](TROUBLESHOOTING.md) or contact the team.
