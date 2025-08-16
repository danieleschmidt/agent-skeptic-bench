#!/bin/bash

# Production Deployment Script for Agent Skeptic Bench
# Autonomous SDLC v4.0 - Complete Production Deployment

set -euo pipefail

# Configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
REGISTRY="${REGISTRY:-your-registry.com/agent-skeptic-bench}"
VERSION="${VERSION:-latest}"
NAMESPACE="${NAMESPACE:-agent-skeptic-bench}"
CLUSTER_NAME="${CLUSTER_NAME:-production-cluster}"
REGION="${REGION:-us-east-1}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local required_tools=("docker" "kubectl" "helm" "jq" "curl")
    local missing_tools=()
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    # Check kubectl context
    if ! kubectl config current-context | grep -q "$CLUSTER_NAME"; then
        log_warning "Current kubectl context doesn't match expected cluster: $CLUSTER_NAME"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_success "Prerequisites check passed"
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    # Main application image
    log_info "Building main application image..."
    docker build -t "${REGISTRY}:${VERSION}" \
        --target production \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --build-arg VERSION="$VERSION" \
        -f deployment/Dockerfile .
    
    # Quantum worker image
    log_info "Building quantum worker image..."
    docker build -t "${REGISTRY}:quantum-worker-${VERSION}" \
        --target quantum-worker \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --build-arg VERSION="$VERSION" \
        -f deployment/Dockerfile .
    
    # Research worker image
    log_info "Building research worker image..."
    docker build -t "${REGISTRY}:research-worker-${VERSION}" \
        --target research-worker \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --build-arg VERSION="$VERSION" \
        -f deployment/Dockerfile .
    
    # Security scanner image
    log_info "Building security scanner image..."
    docker build -t "${REGISTRY}:security-scanner-${VERSION}" \
        --target security-scanner \
        --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
        --build-arg VCS_REF="$(git rev-parse HEAD)" \
        --build-arg VERSION="$VERSION" \
        -f deployment/Dockerfile .
    
    # Push images
    log_info "Pushing images to registry..."
    docker push "${REGISTRY}:${VERSION}"
    docker push "${REGISTRY}:quantum-worker-${VERSION}"
    docker push "${REGISTRY}:research-worker-${VERSION}"
    docker push "${REGISTRY}:security-scanner-${VERSION}"
    
    log_success "Images built and pushed successfully"
}

# Setup monitoring infrastructure
setup_monitoring() {
    log_info "Setting up monitoring infrastructure..."
    
    # Add Prometheus Helm repository
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
    helm repo update
    
    # Create monitoring namespace
    kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
    
    # Install Prometheus
    log_info "Installing Prometheus..."
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi \
        --set alertmanager.alertmanagerSpec.storage.volumeClaimTemplate.spec.resources.requests.storage=10Gi \
        --set grafana.persistence.enabled=true \
        --set grafana.persistence.size=10Gi \
        --wait
    
    # Install Jaeger
    log_info "Installing Jaeger..."
    helm upgrade --install jaeger jaegertracing/jaeger \
        --namespace monitoring \
        --set provisionDataStore.cassandra=false \
        --set provisionDataStore.elasticsearch=true \
        --set storage.type=elasticsearch \
        --wait
    
    log_success "Monitoring infrastructure setup completed"
}

# Deploy secrets and configurations
deploy_secrets() {
    log_info "Deploying secrets and configurations..."
    
    # Generate secrets if they don't exist
    if ! kubectl get secret app-secrets -n "$NAMESPACE" &> /dev/null; then
        log_info "Generating application secrets..."
        
        # Generate random passwords
        DB_PASSWORD=$(openssl rand -base64 32)
        GRAFANA_PASSWORD=$(openssl rand -base64 16)
        JWT_SECRET=$(openssl rand -base64 32)
        
        # Create secret
        kubectl create secret generic app-secrets \
            --namespace="$NAMESPACE" \
            --from-literal=DB_PASSWORD="$DB_PASSWORD" \
            --from-literal=GRAFANA_PASSWORD="$GRAFANA_PASSWORD" \
            --from-literal=JWT_SECRET="$JWT_SECRET"
    else
        log_info "Secrets already exist, skipping generation"
    fi
    
    # Apply ConfigMaps
    kubectl apply -f deployment/kubernetes-production.yaml
    
    log_success "Secrets and configurations deployed"
}

# Deploy application
deploy_application() {
    log_info "Deploying application..."
    
    # Create namespace
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Label namespace for monitoring
    kubectl label namespace "$NAMESPACE" monitoring=enabled --overwrite
    
    # Deploy secrets first
    deploy_secrets
    
    # Update image tags in deployment
    sed -i.bak "s|agent-skeptic-bench:latest|${REGISTRY}:${VERSION}|g" deployment/kubernetes-production.yaml
    sed -i.bak "s|agent-skeptic-bench:quantum-worker|${REGISTRY}:quantum-worker-${VERSION}|g" deployment/kubernetes-production.yaml
    sed -i.bak "s|agent-skeptic-bench:research-worker|${REGISTRY}:research-worker-${VERSION}|g" deployment/kubernetes-production.yaml
    
    # Apply deployment
    kubectl apply -f deployment/kubernetes-production.yaml
    
    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --for=condition=available --timeout=600s deployment/agent-skeptic-bench -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=600s deployment/quantum-worker -n "$NAMESPACE"
    kubectl wait --for=condition=available --timeout=600s deployment/research-worker -n "$NAMESPACE"
    
    # Wait for StatefulSets
    kubectl wait --for=condition=ready --timeout=600s statefulset/postgres -n "$NAMESPACE"
    
    log_success "Application deployed successfully"
}

# Setup ingress and TLS
setup_ingress() {
    log_info "Setting up ingress and TLS..."
    
    # Install nginx-ingress if not present
    if ! kubectl get namespace ingress-nginx &> /dev/null; then
        log_info "Installing nginx-ingress controller..."
        kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
        
        # Wait for ingress controller to be ready
        kubectl wait --namespace ingress-nginx \
            --for=condition=ready pod \
            --selector=app.kubernetes.io/component=controller \
            --timeout=120s
    fi
    
    # Install cert-manager if not present
    if ! kubectl get namespace cert-manager &> /dev/null; then
        log_info "Installing cert-manager..."
        kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
        
        # Wait for cert-manager to be ready
        kubectl wait --namespace cert-manager \
            --for=condition=ready pod \
            --selector=app.kubernetes.io/name=cert-manager \
            --timeout=120s
    fi
    
    # Create ClusterIssuer for Let's Encrypt
    cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@agent-skeptic-bench.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
    
    log_success "Ingress and TLS setup completed"
}

# Run post-deployment tests
run_post_deployment_tests() {
    log_info "Running post-deployment tests..."
    
    # Get service endpoints
    local service_ip
    if kubectl get service agent-skeptic-bench-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' &> /dev/null; then
        service_ip=$(kubectl get service agent-skeptic-bench-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    else
        service_ip=$(kubectl get service agent-skeptic-bench-service -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    fi
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    if kubectl run test-pod --rm -i --restart=Never --image=curlimages/curl -- \
        curl -f "http://${service_ip}/health" &> /dev/null; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi
    
    # Test metrics endpoint
    log_info "Testing metrics endpoint..."
    if kubectl run test-pod --rm -i --restart=Never --image=curlimages/curl -- \
        curl -f "http://${service_ip}:8001/metrics" &> /dev/null; then
        log_success "Metrics endpoint accessible"
    else
        log_warning "Metrics endpoint not accessible"
    fi
    
    # Test database connectivity
    log_info "Testing database connectivity..."
    if kubectl exec -n "$NAMESPACE" statefulset/postgres -- pg_isready -U skeptic -d skeptic_bench; then
        log_success "Database connectivity test passed"
    else
        log_error "Database connectivity test failed"
        return 1
    fi
    
    # Test Redis connectivity
    log_info "Testing Redis connectivity..."
    if kubectl exec -n "$NAMESPACE" deployment/redis -- redis-cli ping | grep -q PONG; then
        log_success "Redis connectivity test passed"
    else
        log_error "Redis connectivity test failed"
        return 1
    fi
    
    log_success "All post-deployment tests passed"
}

# Generate deployment report
generate_deployment_report() {
    log_info "Generating deployment report..."
    
    local report_file="deployment-report-$(date +%Y%m%d-%H%M%S).txt"
    
    cat > "$report_file" << EOF
# Agent Skeptic Bench - Production Deployment Report
Generated: $(date)
Environment: $DEPLOYMENT_ENV
Version: $VERSION
Namespace: $NAMESPACE
Cluster: $CLUSTER_NAME

## Deployment Status
$(kubectl get deployments -n "$NAMESPACE" -o wide)

## Pod Status
$(kubectl get pods -n "$NAMESPACE" -o wide)

## Service Status
$(kubectl get services -n "$NAMESPACE" -o wide)

## Ingress Status
$(kubectl get ingress -n "$NAMESPACE" -o wide)

## HPA Status
$(kubectl get hpa -n "$NAMESPACE" -o wide)

## Resource Usage
$(kubectl top pods -n "$NAMESPACE" 2>/dev/null || echo "Metrics not available")

## Storage
$(kubectl get pvc -n "$NAMESPACE")

## Events (Last 10)
$(kubectl get events -n "$NAMESPACE" --sort-by='.lastTimestamp' | tail -10)

## Application Logs (Last 50 lines)
$(kubectl logs -n "$NAMESPACE" deployment/agent-skeptic-bench --tail=50)

EOF
    
    log_success "Deployment report generated: $report_file"
}

# Rollback function
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    kubectl rollout undo deployment/agent-skeptic-bench -n "$NAMESPACE"
    kubectl rollout undo deployment/quantum-worker -n "$NAMESPACE"
    kubectl rollout undo deployment/research-worker -n "$NAMESPACE"
    
    # Wait for rollback to complete
    kubectl rollout status deployment/agent-skeptic-bench -n "$NAMESPACE"
    kubectl rollout status deployment/quantum-worker -n "$NAMESPACE"
    kubectl rollout status deployment/research-worker -n "$NAMESPACE"
    
    log_success "Rollback completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f deployment/kubernetes-production.yaml.bak
}

# Main deployment function
main() {
    log_info "Starting production deployment for Agent Skeptic Bench v$VERSION"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Check if this is a rollback
    if [ "${1:-}" = "rollback" ]; then
        rollback_deployment
        exit 0
    fi
    
    # Run deployment steps
    check_prerequisites
    
    if [ "${SKIP_BUILD:-false}" != "true" ]; then
        build_and_push_images
    else
        log_info "Skipping image build (SKIP_BUILD=true)"
    fi
    
    if [ "${SKIP_MONITORING:-false}" != "true" ]; then
        setup_monitoring
    else
        log_info "Skipping monitoring setup (SKIP_MONITORING=true)"
    fi
    
    deploy_application
    
    if [ "${SKIP_INGRESS:-false}" != "true" ]; then
        setup_ingress
    else
        log_info "Skipping ingress setup (SKIP_INGRESS=true)"
    fi
    
    # Wait a bit for services to stabilize
    log_info "Waiting for services to stabilize..."
    sleep 30
    
    if [ "${SKIP_TESTS:-false}" != "true" ]; then
        if ! run_post_deployment_tests; then
            log_error "Post-deployment tests failed!"
            read -p "Continue anyway? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Rolling back due to test failures..."
                rollback_deployment
                exit 1
            fi
        fi
    else
        log_info "Skipping post-deployment tests (SKIP_TESTS=true)"
    fi
    
    generate_deployment_report
    
    log_success "Production deployment completed successfully!"
    log_info "Application is available at: http://api.agent-skeptic-bench.com"
    log_info "Monitoring dashboards:"
    log_info "  - Grafana: http://grafana.agent-skeptic-bench.com"
    log_info "  - Prometheus: http://prometheus.agent-skeptic-bench.com"
    log_info "  - Jaeger: http://jaeger.agent-skeptic-bench.com"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "rollback")
        main rollback
        ;;
    "test")
        check_prerequisites
        run_post_deployment_tests
        ;;
    "report")
        generate_deployment_report
        ;;
    *)
        echo "Usage: $0 {deploy|rollback|test|report}"
        exit 1
        ;;
esac