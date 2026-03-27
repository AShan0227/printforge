#!/bin/bash
# PrintForge v2.2 — One-click deploy
set -e

echo "🚀 PrintForge v2.2.0 Deploy"
echo ""

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "❌ python3 required"; exit 1; }
command -v pip >/dev/null 2>&1 || command -v uv >/dev/null 2>&1 || { echo "❌ pip or uv required"; exit 1; }

cd "$(dirname "$0")/.."
ROOT=$(pwd)

echo "[1/5] Installing dependencies..."
if command -v uv >/dev/null 2>&1; then
    uv pip install -e ".[dev]" 2>/dev/null || uv pip install -e . 2>/dev/null || pip install -e .
else
    pip install -e .
fi

echo "[2/5] Installing TripoSR dependencies..."
pip install transformers omegaconf einops "rembg[cpu]" scikit-image 2>/dev/null || true

echo "[3/5] Downloading TripoSR model (if not cached)..."
python3 -c "
from huggingface_hub import snapshot_download
import os
cache = os.path.expanduser('~/.cache/huggingface/hub/models--stabilityai--TripoSR')
if os.path.isdir(cache):
    print('  Model already cached')
else:
    print('  Downloading TripoSR (~1.6GB)...')
    snapshot_download('stabilityai/TripoSR')
    print('  Done')
" 2>/dev/null || echo "  ⚠️ Model download skipped (will download on first use)"

echo "[4/5] Running tests..."
python3 -m pytest tests/test_api_v2.py -v --tb=short 2>/dev/null || echo "  ⚠️ Some tests failed (non-blocking)"

echo "[5/5] Starting server..."
echo ""
echo "═══════════════════════════════════════════════════"
echo "  PrintForge v2.2.0 — Ready"
echo ""
echo "  🌐 API:     http://localhost:8000/docs"
echo "  🎨 Preview: http://localhost:8000/preview"
echo "  📊 Metrics: http://localhost:8000/metrics"
echo "  ❤️  Health:  http://localhost:8000/health/detail"
echo ""
echo "  Quick start:"
echo "    printforge register myuser"
echo "    printforge image photo.png -o model.stl"
echo "    printforge serve"
echo "═══════════════════════════════════════════════════"
echo ""

exec uvicorn printforge.server:app --host 0.0.0.0 --port 8000
