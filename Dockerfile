FROM python:3.12-slim AS base

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        python3-dev \
        gcc \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (cache layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . 2>/dev/null || pip install --no-cache-dir .

# Copy source
COPY src/ src/
COPY web/ web/
COPY examples/ examples/

# Download TripoSR for local inference (optional, makes image larger but faster first run)
# RUN pip install --no-cache-dir transformers omegaconf einops && \
#     python -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/TripoSR')"

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "printforge.server:app", "--host", "0.0.0.0", "--port", "8000"]
