"""
gunicorn.conf.py — Gunicorn configuration for production deployment.

CPU-bound ML workloads need fewer workers than typical web apps.
Preload the app to share model memory across workers.
"""

import multiprocessing

# ── Server Socket ───────────────────────────────────────────────
bind = "0.0.0.0:5000"

# ── Workers ─────────────────────────────────────────────────────
# For ML workloads: use 2 workers (models consume significant memory)
workers = min(2, multiprocessing.cpu_count())
worker_class = "sync"
threads = 1

# ── Timeouts ────────────────────────────────────────────────────
timeout = 120        # Model loading can take time
graceful_timeout = 30
keepalive = 5

# ── Preload ─────────────────────────────────────────────────────
# Preload app so models are loaded once and shared across workers
preload_app = True

# ── Logging ─────────────────────────────────────────────────────
accesslog = "logs/gunicorn_access.log"
errorlog = "logs/gunicorn_error.log"
loglevel = "info"

# ── Request Limits ──────────────────────────────────────────────
limit_request_line = 8190
limit_request_fields = 100
limit_request_field_size = 8190
