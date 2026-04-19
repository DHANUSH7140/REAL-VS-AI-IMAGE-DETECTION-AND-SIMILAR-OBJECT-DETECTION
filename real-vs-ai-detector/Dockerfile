# ── Stage 1: Build dependencies ────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Production image ──────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV + general
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p static/uploads static/gradcam static/fft static/similarity \
    static/patches static/evaluation model/meta model/onnx logs

# Expose port
EXPOSE 5000

# Environment variables
ENV FLASK_DEBUG=false
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:5000/api/health'); exit(0 if r.ok else 1)" || exit 1

# Run with Gunicorn
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:create_app()"]
