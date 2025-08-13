#!/bin/bash
set -euo pipefail

# Agent Skeptic Bench Rollback Script
# Usage: ./rollback.sh [environment] [revision]

ENVIRONMENT=${1:-production}
REVISION=${2:-}
NAMESPACE="agent-skeptic-bench-${ENVIRONMENT}"

echo "🔄 Rolling back Agent Skeptic Bench in ${ENVIRONMENT}"

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

# Set kubectl context
kubectl config use-context "agent-skeptic-bench-${ENVIRONMENT}"

if [ -z "$REVISION" ]; then
    echo "🔍 Finding previous revision"
    REVISION=$(kubectl rollout history deployment/agent-skeptic-bench -n $NAMESPACE | tail -2 | head -1 | awk '{print $1}')
    echo "📝 Rolling back to revision: $REVISION"
fi

# Perform rollback
echo "🔄 Performing rollback"
kubectl rollout undo deployment/agent-skeptic-bench -n $NAMESPACE --to-revision=$REVISION

# Wait for rollback to complete
echo "⏳ Waiting for rollback to complete"
kubectl rollout status deployment/agent-skeptic-bench -n $NAMESPACE --timeout=300s

# Verify rollback
echo "🔍 Verifying rollback"
kubectl get pods -n $NAMESPACE -l app=agent-skeptic-bench

# Run health check
echo "❤️  Running health check"
kubectl port-forward -n $NAMESPACE service/agent-skeptic-bench-service 8080:80 &
PF_PID=$!
sleep 10

if curl -f http://localhost:8080/health; then
    echo "✅ Rollback successful - health check passed"
else
    echo "❌ Rollback failed - health check failed"
    kill $PF_PID 2>/dev/null || true
    exit 1
fi

kill $PF_PID 2>/dev/null || true

echo "🎉 Rollback completed successfully!"
echo "📊 Monitor at: https://grafana.agent-skeptic-bench.com"
