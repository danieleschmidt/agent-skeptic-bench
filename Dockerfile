# Multi-stage build for optimized production image
FROM python:3.13-slim as builder

# Set build arguments
ARG BUILD_ENV=production
ARG VERSION=1.0.0

# Install build dependencies and security tools
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies based on environment
RUN if [ "$BUILD_ENV" = "development" ]; then \
        pip install --no-cache-dir -e .[dev,monitoring,security,docs,test]; \
    else \
        pip install --no-cache-dir -e .[monitoring,security]; \
    fi

# Production stage
FROM python:3.13-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_ENV=production

# Install runtime dependencies and security packages
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    libpq5 \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash --uid 1000 app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/results /app/certs \
    && chown -R app:app /app

# Copy application code
COPY --chown=app:app src/ /app/src/
COPY --chown=app:app docs/ /app/docs/
COPY --chown=app:app monitoring/ /app/monitoring/
COPY --chown=app:app pyproject.toml /app/

WORKDIR /app

# Install the application
RUN pip install --no-cache-dir -e .

# Switch to non-root user
USER app

# Expose ports
EXPOSE 8000 9090

# Health check with comprehensive checks
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import agent_skeptic_bench; from agent_skeptic_bench.api.app import create_app; print('healthy')" || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command (can be overridden)
CMD ["python", "-m", "agent_skeptic_bench.cli", "--help"]

# Development stage (optional)
FROM production as development

# Install development dependencies
USER root
RUN apt-get update && apt-get install -y \
    vim \
    git \
    htop \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

USER app

# Copy development configuration
COPY --chown=app:app .env.example /app/.env.example

# Development command
CMD ["python", "-m", "agent_skeptic_bench.api.app"]

# Labels for metadata
LABEL maintainer="skeptic-bench@yourdomain.com" \
      version="1.0.0" \
      description="Agent Skeptic Bench - Epistemic Vigilance Evaluation" \
      org.opencontainers.image.title="Agent Skeptic Bench" \
      org.opencontainers.image.description="AI Agent Skepticism Evaluation Framework" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/danieleschmidt/agent-skeptic-bench"
