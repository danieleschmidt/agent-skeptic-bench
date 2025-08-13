#!/bin/bash
set -euo pipefail

# Agent Skeptic Bench Production Deployment Script
# Usage: ./deploy.sh [environment] [version]

ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
NAMESPACE="agent-skeptic-bench-${ENVIRONMENT}"

echo "🚀 Deploying Agent Skeptic Bench v${VERSION} to ${ENVIRONMENT}"

# Validate environment
case $ENVIRONMENT in
  staging|production)
    echo "✅ Environment: $ENVIRONMENT"
    ;;
  *)
    echo "❌ Invalid environment. Use 'staging' or 'production'"
    exit 1
    ;;
esac

# Check prerequisites
command -v kubectl >/dev/null 2>&1 || { echo "❌ kubectl not found"; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "❌ docker not found"; exit 1; }

# Set kubectl context
echo "📋 Setting kubectl context for $ENVIRONMENT"
kubectl config use-context "agent-skeptic-bench-${ENVIRONMENT}"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply secrets (if not exist)
echo "🔐 Applying secrets"
kubectl apply -f kubernetes-secrets.yaml -n $NAMESPACE

# Apply configmaps
echo "📝 Applying configuration"
kubectl apply -f kubernetes-configmap.yaml -n $NAMESPACE

# Deploy application
echo "🚀 Deploying application"
kubectl apply -f kubernetes-deployment.yaml -n $NAMESPACE
kubectl apply -f kubernetes-service.yaml -n $NAMESPACE
kubectl apply -f kubernetes-ingress.yaml -n $NAMESPACE

# Wait for rollout
echo "⏳ Waiting for deployment to complete"
kubectl rollout status deployment/agent-skeptic-bench -n $NAMESPACE --timeout=600s

# Verify deployment
echo "🔍 Verifying deployment"
kubectl get pods -n $NAMESPACE -l app=agent-skeptic-bench
kubectl get services -n $NAMESPACE

# Run health check
echo "❤️  Running health check"
kubectl port-forward -n $NAMESPACE service/agent-skeptic-bench-service 8080:80 &
PF_PID=$!
sleep 10

if curl -f http://localhost:8080/health; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    kill $PF_PID 2>/dev/null || true
    exit 1
fi

kill $PF_PID 2>/dev/null || true

echo "🎉 Deployment completed successfully!"
echo "📊 Monitor at: https://grafana.agent-skeptic-bench.com"
echo "🔗 API URL: https://api.agent-skeptic-bench.com"
