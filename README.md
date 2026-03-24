# 🏭 PrintForge — One Photo to 3D Print

[![CI](https://github.com/YOUR_USERNAME/printforge/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/printforge/actions/workflows/ci.yml)
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Open-source, local-first AI pipeline that turns a single photo or text description into a print-ready 3D model.**

<!-- ![demo](docs/demo.gif) -->

---

## Features

- 📸 **Image to 3D** — Drop in a photo, get a watertight mesh in seconds
- 💬 **Text to 3D** — Describe what you want, AI generates the model
- 🔧 **Print Optimizer** — Auto-orientation, time/material estimation, printability analysis
- ✂️ **Part Splitter** — Automatically split oversized models with alignment pins
- 🖨️ **Bambu Lab Integration** — Discover and send prints directly to Bambu printers via MQTT/FTP
- 🌐 **Web UI** — Modern dark-theme interface with real-time progress and 3D preview
- 🔌 **REST API** — FastAPI server with WebSocket progress streaming
- 📦 **Multi-format Export** — 3MF, STL (binary/ASCII), OBJ with materials

---

## Install

### pip (recommended)

```bash
pip install printforge
```

### From source

```bash
git clone https://github.com/YOUR_USERNAME/printforge.git
cd printforge
pip install -e '.[dev]'
```

### Docker

```bash
docker compose up --build
# Web UI at http://localhost:8000
```

### One-liner

```bash
curl -sSL https://raw.githubusercontent.com/YOUR_USERNAME/printforge/main/scripts/install.sh | bash
```

---

## Usage

### CLI

```bash
# Image to 3D
printforge image photo.jpg -o model.3mf

# Text to 3D
printforge text "a small gear with 12 teeth" -o gear.stl

# Analyze print settings
printforge optimize model.stl --printer bambu-a1

# Split oversized model
printforge split large_model.stl --volume 256x256x256
```

### Python API

```python
from printforge.pipeline import PrintForgePipeline, PipelineConfig

config = PipelineConfig(resolution=256, export_format="3mf")
pipeline = PrintForgePipeline(config)
result = pipeline.run("photo.jpg", output_path="model.3mf")
print(f"Vertices: {result['vertices']}, Watertight: {result['watertight']}")
```

### Web UI

```bash
printforge server
# or
uvicorn printforge.server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 — drag-drop an image or type a description.

### REST API

```bash
# Generate from image
curl -X POST http://localhost:8000/api/generate \
  -F "file=@photo.jpg" -F "format=3mf" -o model.3mf

# Generate from text
curl -X POST http://localhost:8000/api/text-to-3d \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a small vase", "format": "stl"}'

# Analyze printability
curl -X POST http://localhost:8000/api/optimize -F "file=@model.stl"

# List formats & printer presets
curl http://localhost:8000/api/formats
```

---

## Bambu Lab Integration

PrintForge discovers Bambu Lab printers on your network via SSDP and can send prints directly.

```bash
# Discover printers
curl http://localhost:8000/api/printers

# Send to printer
curl -X POST http://localhost:8000/api/printers/send \
  -F "file=@model.3mf" -F "printer_ip=192.168.1.100" -F "access_code=YOUR_CODE"
```

Supported printers: X1 Carbon, X1, P1S, P1P, A1, A1 Mini.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Photo/Text │────▶│   TripoSR    │────▶│  SDF Field   │
│   Input     │     │  Inference   │     │  Watertight  │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                │
                    ┌──────────────┐     ┌──────▼───────┐
                    │  3MF / STL   │◀────│   Marching   │
                    │   Export     │     │    Cubes     │
                    └──────┬───────┘     └──────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │   CLI    │ │  Web UI  │ │ REST API │
        └──────────┘ └──────────┘ └──────────┘
```

---

## Supported Formats

| Format | Extension | Features |
|--------|-----------|----------|
| 3MF    | `.3mf`    | Metadata, print settings, multi-material |
| STL    | `.stl`    | Binary (default) and ASCII modes |
| OBJ    | `.obj`    | With `.mtl` material file |

---

## Roadmap

### P0 — Core (done)
- [x] Image to 3D pipeline with watertight guarantee
- [x] CLI, Web UI, REST API
- [x] Print optimizer & part splitter
- [x] Bambu Lab printer integration
- [x] Multi-format export (3MF/STL/OBJ)

### P1 — Enhanced
- [ ] Multi-view enhancement (Zero123++/SV3D)
- [ ] Support structure optimizer
- [ ] Material advisor with cost estimation
- [ ] Batch processing

### P2 — Platform
- [ ] Model marketplace & templates
- [ ] Sketch to 3D
- [ ] Strength analysis (FEA)
- [ ] Print failure prediction

---

## Contributing

```bash
git clone https://github.com/YOUR_USERNAME/printforge.git
cd printforge
pip install -e '.[dev]'
pytest tests/ -v
```

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Write tests first, then implement
4. Run `pytest tests/ -v` and `ruff check src/`
5. Open a PR

---

## License

MIT — see [LICENSE](LICENSE) for details.
