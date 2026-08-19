FROM python:3.12-slim AS base

# Build arguments — override at build time if needed.
ARG PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PICTOGRAM_CACHE_DIR=/app/cache

WORKDIR /app

# ---------------------------------------------------------------------------
# 1. Install system dependencies (needed by Pillow / numpy)
# ---------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libjpeg62-turbo \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. Install Python dependencies (cached layer)
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# 3. Download and cache the dataset during build
#    This means containers start instantly — no Hugging Face fetch at runtime.
# ---------------------------------------------------------------------------
COPY scripts/ ./scripts/
RUN python -m scripts.download_and_cache --output-dir /app/cache

# ---------------------------------------------------------------------------
# 4. Copy application code (changes frequently — placed last for cache)
# ---------------------------------------------------------------------------
COPY src/ ./src/
COPY main.py .

# ---------------------------------------------------------------------------
# 5. Non-root user for security
# ---------------------------------------------------------------------------
RUN groupadd --system app && \
    useradd --system --gid app app && \
    chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "6027", "--workers", "2"]
