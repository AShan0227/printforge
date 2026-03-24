#!/usr/bin/env bash
set -euo pipefail

MIN_PYTHON="3.9"

echo "🏭 PrintForge Installer"
echo "======================="

# Check python3 exists
if ! command -v python3 &>/dev/null; then
    echo "❌ python3 not found. Please install Python $MIN_PYTHON or later."
    exit 1
fi

# Check version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]; }; then
    echo "❌ Python $MIN_PYTHON+ required, found $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Install
echo "📦 Installing PrintForge..."
pip install printforge

echo ""
echo "✅ PrintForge installed successfully!"
echo ""
echo "Quick start:"
echo "  printforge image photo.jpg -o model.3mf    # Image to 3D"
echo "  printforge text 'a small vase' -o vase.stl  # Text to 3D"
echo "  printforge optimize model.stl                # Analyze print"
echo "  uvicorn printforge.server:app --port 8000    # Web UI"
echo ""
echo "Docs: https://github.com/YOUR_USERNAME/printforge"
