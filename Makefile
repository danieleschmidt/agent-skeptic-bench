# Makefile for Agent Skeptic Bench

# Variables
PYTHON := python
PIP := pip
PROJECT_NAME := agent-skeptic-bench
VERSION := $(shell grep '^version' pyproject.toml | cut -d'"' -f2)
DOCKER_IMAGE := $(PROJECT_NAME):$(VERSION)
DOCKER_REGISTRY := ghcr.io/danieleschmidt
DOCKER_TAG := latest

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: help install install-dev test lint format type-check security-check clean build docker-build docker-run docker-push docs serve-docs benchmark setup-dev deps-check health-check

# Default target
help: ## Show this help message
	@echo "$(BLUE)Agent Skeptic Bench - Build Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# Installation targets
install: ## Install the package in production mode
	@echo "$(BLUE)Installing Agent Skeptic Bench...$(NC)"
	$(PIP) install -e .

install-dev: ## Install the package in development mode with all dependencies
	@echo "$(BLUE)Installing Agent Skeptic Bench in development mode...$(NC)"
	$(PIP) install -e .[dev,monitoring,security,docs,test]
	pre-commit install

deps-check: ## Check for dependency vulnerabilities
	@echo "$(BLUE)Checking dependencies for vulnerabilities...$(NC)"
	safety check
	pip-audit

# Development targets
setup-dev: install-dev ## Set up development environment
	@echo "$(GREEN)Development environment setup complete!$(NC)"
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and configure"
	@echo "  2. Start services: make docker-compose-up"
	@echo "  3. Run tests: make test"

clean: ## Clean build artifacts and caches
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

# Code quality targets
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ tests/
	isort src/ tests/

lint: ## Lint code with ruff
	@echo "$(BLUE)Linting code...$(NC)"
	ruff check src/ tests/ --fix

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Type checking...$(NC)"
	mypy src/

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	bandit -r src/
	semgrep --config=auto src/

code-quality: format lint type-check security-check ## Run all code quality checks

# Testing targets
test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/unit/ -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/integration/ -v

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	pytest tests/performance/ -v

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ --cov=agent_skeptic_bench --cov-report=html --cov-report=term

# Build targets
build: clean ## Build the package
	@echo "$(BLUE)Building package...$(NC)"
	$(PYTHON) -m build

# Docker targets
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE) -t $(DOCKER_REGISTRY)/$(PROJECT_NAME):$(VERSION) -t $(DOCKER_REGISTRY)/$(PROJECT_NAME):$(DOCKER_TAG) .

docker-build-dev: ## Build Docker image for development
	@echo "$(BLUE)Building Docker development image...$(NC)"
	docker build --target development -t $(DOCKER_IMAGE)-dev .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm \
		--env-file .env \
		-p 8000:8000 \
		-p 9090:9090 \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/results:/app/results \
		$(DOCKER_IMAGE)

docker-run-dev: ## Run Docker container in development mode
	@echo "$(BLUE)Running Docker development container...$(NC)"
	docker run -it --rm \
		--env-file .env \
		-p 8000:8000 \
		-p 9090:9090 \
		-v $(PWD):/app \
		$(DOCKER_IMAGE)-dev

docker-push: docker-build ## Push Docker image to registry
	@echo "$(BLUE)Pushing Docker image...$(NC)"
	docker push $(DOCKER_REGISTRY)/$(PROJECT_NAME):$(VERSION)
	docker push $(DOCKER_REGISTRY)/$(PROJECT_NAME):$(DOCKER_TAG)

docker-compose-up: ## Start services with docker-compose
	@echo "$(BLUE)Starting services with docker-compose...$(NC)"
	docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	@echo "$(BLUE)Stopping services with docker-compose...$(NC)"
	docker-compose down

docker-compose-logs: ## View docker-compose logs
	docker-compose logs -f

# Documentation targets
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd docs && mkdocs build

serve-docs: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8080$(NC)"
	cd docs && mkdocs serve -a 0.0.0.0:8080

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(BLUE)Deploying documentation...$(NC)"
	cd docs && mkdocs gh-deploy

# Application targets
benchmark: ## Run sample benchmark evaluation
	@echo "$(BLUE)Running sample benchmark...$(NC)"
	agent-skeptic-bench evaluate --model gpt-4 --categories factual_claims --limit 5

serve: ## Start the API server
	@echo "$(BLUE)Starting API server at http://localhost:8000$(NC)"
	uvicorn agent_skeptic_bench.api.app:create_production_app --host 0.0.0.0 --port 8000 --reload

serve-prod: ## Start the API server in production mode
	@echo "$(BLUE)Starting API server in production mode...$(NC)"
	gunicorn agent_skeptic_bench.api.app:create_production_app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Database targets
db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	$(PYTHON) -c "from agent_skeptic_bench.database.connection import get_database; import asyncio; asyncio.run(get_database().create_tables())"

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "$(RED)WARNING: This will destroy all data!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(PYTHON) -c "from agent_skeptic_bench.database.connection import get_database; import asyncio; asyncio.run(get_database().drop_tables()); asyncio.run(get_database().create_tables())"; \
		echo "$(GREEN)Database reset complete$(NC)"; \
	else \
		echo "$(YELLOW)Database reset cancelled$(NC)"; \
	fi

db-backup: ## Backup database
	@echo "$(BLUE)Creating database backup...$(NC)"
	pg_dump $(DATABASE_URL) > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Health check targets
health-check: ## Check application health
	@echo "$(BLUE)Checking application health...$(NC)"
	curl -f http://localhost:8000/health || echo "$(RED)Health check failed$(NC)"

health-check-all: ## Check all services health
	@echo "$(BLUE)Checking all services...$(NC)"
	@echo "API Health:"
	@curl -f http://localhost:8000/health 2>/dev/null | jq . || echo "$(RED)API health check failed$(NC)"
	@echo "Database Health:"
	@$(PYTHON) -c "from agent_skeptic_bench.database.connection import get_database; import asyncio; print('Database:', asyncio.run(get_database().health_check()))" || echo "$(RED)Database health check failed$(NC)"

# Performance targets
profile: ## Profile application performance
	@echo "$(BLUE)Profiling application...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats -m agent_skeptic_bench.cli benchmark --model test --limit 10
	$(PYTHON) -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

load-test: ## Run load tests
	@echo "$(BLUE)Running load tests...$(NC)"
	locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Release targets
version: ## Show current version
	@echo "Current version: $(VERSION)"

bump-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(NC)"
	bump2version patch

bump-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(NC)"
	bump2version minor

bump-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(NC)"
	bump2version major

release: test build docker-build ## Create a release (run tests, build, and create Docker image)
	@echo "$(GREEN)Release $(VERSION) ready!$(NC)"
	@echo "Next steps:"
	@echo "  1. Push Docker image: make docker-push"
	@echo "  2. Create GitHub release"
	@echo "  3. Deploy to production"

# Utility targets
logs: ## View application logs
	tail -f logs/agent_skeptic_bench.log

monitor: ## Start monitoring dashboard
	@echo "$(BLUE)Starting monitoring dashboard...$(NC)"
	docker-compose up -d prometheus grafana

backup: ## Create backup of data and results
	@echo "$(BLUE)Creating backup...$(NC)"
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/ results/ logs/

restore: ## Restore from backup (specify BACKUP_FILE)
	@echo "$(BLUE)Restoring from $(BACKUP_FILE)...$(NC)"
	tar -xzf $(BACKUP_FILE)

# Pre-commit hooks
pre-commit: format lint type-check test ## Run all pre-commit checks

# CI/CD targets
ci-test: ## Run CI test suite
	@echo "$(BLUE)Running CI test suite...$(NC)"
	pytest tests/ --cov=agent_skeptic_bench --cov-report=xml --junitxml=test-results.xml

ci-build: ## Build for CI
	@echo "$(BLUE)Building for CI...$(NC)"
	$(MAKE) clean build docker-build

ci-deploy: ## Deploy for CI
	@echo "$(BLUE)Deploying for CI...$(NC)"
	$(MAKE) docker-push docs-deploy

# Environment info
env-info: ## Show environment information
	@echo "$(BLUE)Environment Information:$(NC)"
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Pip version: $(shell $(PIP) --version)"
	@echo "Project version: $(VERSION)"
	@echo "Docker version: $(shell docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Working directory: $(PWD)"