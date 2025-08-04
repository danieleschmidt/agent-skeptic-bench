# Multi-stage production Dockerfile for Agent Skeptic Bench
FROM python:3.12-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Install additional production dependencies
RUN pip install --no-cache-dir \
    gunicorn \
    uvicorn[standard] \
    prometheus-client \
    psutil \
    redis \
    aioredis \
    psycopg2-binary \
    structlog

# Production stage
FROM python:3.12-slim as production

# Create non-root user
RUN groupadd -r asb && useradd -r -g asb asb

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=asb:asb src/ ./src/
COPY --chown=asb:asb data/ ./data/
COPY --chown=asb:asb deployment/entrypoint.sh ./
COPY --chown=asb:asb deployment/gunicorn.conf.py ./

# Create necessary directories
RUN mkdir -p /app/logs /app/cache && \
    chown -R asb:asb /app/logs /app/cache

# Make entrypoint executable
RUN chmod +x ./entrypoint.sh

# Switch to non-root user
USER asb

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]