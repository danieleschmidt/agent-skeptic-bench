"""Gunicorn configuration for Agent Skeptic Bench production deployment."""

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Timeout
timeout = 60
keepalive = 5

# Logging
loglevel = os.getenv("LOG_LEVEL", "info").lower()
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "agent-skeptic-bench"

# Server mechanics
daemon = False
pidfile = "/tmp/gunicorn.pid"
user = None
group = None
tmp_upload_dir = None

# SSL (if certificates are provided)
keyfile = os.getenv("SSL_KEYFILE")
certfile = os.getenv("SSL_CERTFILE")

# Worker lifecycle
preload_app = True
reload = os.getenv("ENVIRONMENT", "production") == "development"

# Application-specific settings
raw_env = [
    f"PYTHONPATH=/app/src",
    f"MAX_CONCURRENT_EVALUATIONS={os.getenv('MAX_CONCURRENT_EVALUATIONS', '16')}",
    f"CACHE_TTL={os.getenv('CACHE_TTL', '3600')}",
    f"ENABLE_PERFORMANCE_MONITORING={os.getenv('ENABLE_PERFORMANCE_MONITORING', 'true')}",
]


def when_ready(server):
    """Called just after the server is started."""
    server.log.info("🚀 Agent Skeptic Bench server is ready to accept connections")


def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    worker.log.info(f"Worker {worker.pid} received INT or QUIT signal")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info(f"Worker {worker.age} about to be forked")


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")


def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forked child, re-executing.")


def when_ready(server):
    """Called just after the server is started."""
    server.log.info("Server is ready. Spawning workers")


def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.info(f"Worker {worker.pid} received SIGABRT signal")


def on_exit(server):
    """Called just before exiting."""
    server.log.info("🛑 Agent Skeptic Bench server is shutting down")