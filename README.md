```
    ____       _       __  ______
   / __ \_____(_)___  / /_/ ____/___  _________ ____
  / /_/ / ___/ / __ \/ __/ /_  / __ \/ ___/ __ `/ _ \
 / ____/ /  / / / / / /_/ __/ / /_/ / /  / /_/ /  __/
/_/   /_/  /_/_/ /_/\__/_/    \____/_/   \__, /\___/
                                        /____/
```

# PrintForge — One Photo to 3D Print

[![CI](https://github.com/printforge/printforge/actions/workflows/ci.yml/badge.svg)](https://github.com/printforge/printforge/actions)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/printforge/printforge?style=social)](https://github.com/printforge/printforge)

**Open-source, local-first AI pipeline that turns a single photo or text description into a print-ready 3D model. No cloud required. No subscription. Just your photo and your printer.**

---

## Why PrintForge?

| | The Problem | PrintForge |
|---|---|---|
| **Local-first** | Cloud services charge per generation and own your data | Runs 100% on your machine — your models stay yours |
| **Print-ready output** | Most AI 3D tools make meshes for rendering, not printing | Every mesh is watertight, wall-checked, and slicer-ready |
| **End-to-end** | Scan → repair → slice → print requires 4+ tools | One pipeline from photo to printer, including Bambu Lab direct send |

---

## Quick Start

```bash
pip install printforge                  # Install
printforge image photo.jpg -o model.3mf # Generate
# Open model.3mf in Bambu Studio / PrusaSlicer / Cura and print!
```

---

## Features

| Feature | Description |
|---------|-------------|
| 📸 **Image to 3D** | Drop in a photo, get a watertight mesh in seconds |
| 💬 **Text to 3D** | Describe what you want — AI generates the model |
| 🎥 **Video to 3D** | Extract frames from video for multi-view reconstruction |
| 🔧 **Print Optimizer** | Auto-orientation, time/material estimation, printability analysis |
| 💰 **Cost Estimator** | Filament, electricity, and total cost breakdown |
| ✂️ **Part Splitter** | Split oversized models with alignment pins & holes |
| 🔍 **Quality Scorer** | 0-100 score with per-metric breakdown |
| 🩹 **Mesh Repair** | Fix holes, non-manifold edges, make watertight |
| ⚠️ **Failure Predictor** | Detect thin walls, overhangs, topple risks before printing |
| 📦 **Batch Processing** | Process entire directories of images in parallel |
| 🖨️ **Bambu Lab Integration** | Discover and send prints directly via MQTT/FTP |
| 🌐 **Web UI** | Dark-theme interface with real-time progress and 3D preview |
| 🔌 **REST API** | FastAPI server with WebSocket progress streaming |
| 📊 **Analytics** | Local usage telemetry (no PII, no network) |

---

## Architecture

```
                    ┌─────────────────────────────┐
                    │         INPUT LAYER          │
                    │  Photo  Text  Video  LiDAR   │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │      INFERENCE ENGINE        │
                    │  TRELLIS / Tripo / TripoSR   │
                    │  SDF → Marching Cubes        │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
    ┌─────────▼──────┐ ┌─────────▼──────┐ ┌─────────▼──────┐
    │   OPTIMIZE     │ │    QUALITY     │ │    EXPORT      │
    │  Orientation   │ │  Score 0-100   │ │  3MF/STL/OBJ   │
    │  Cost estimate │ │  Failure pred  │ │  Watertight ✓   │
    │  Part split    │ │  Mesh repair   │ │  Wall check ✓   │
    └────────────────┘ └────────────────┘ └───────┬────────┘
                                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
    ┌─────────▼──────┐ ┌─────────▼──────┐ ┌─────────▼──────┐
    │     CLI        │ │    WEB UI      │ │   REST API     │
    │  20 commands   │ │  Dark theme    │ │  FastAPI + WS   │
    │  Pipe-friendly │ │  3D preview    │ │  Rate limited   │
    └────────────────┘ └────────────────┘ └────────────────┘
```

---

## Comparison

| Feature | PrintForge | Meshy.ai | Tripo AI | Hitem3D |
|---------|-----------|----------|----------|---------|
| Open source | Yes (MIT) | No | No | No |
| Local/offline | Yes | No | No | No |
| Print-ready output | Yes | Partial | No | Partial |
| Watertight guarantee | Yes | No | No | No |
| Cost estimation | Yes | No | No | No |
| Failure prediction | Yes | No | No | No |
| Printer integration | Bambu Lab | No | No | No |
| Batch processing | Yes | API only | No | No |
| Free tier | Unlimited | 5/day | 3/day | 10/day |
| Price | Free forever | $20/mo | $50/mo | $10/mo |

---

## CLI Reference

```
printforge <command> [options]

GENERATION
  image <path>          Convert image to 3D model
  text <description>    Generate 3D from text description
  video <path>          Convert video to 3D via frame extraction

OPTIMIZATION
  optimize <mesh>       Print analysis, orientation, estimates
  cost <mesh>           Cost estimation (filament, time, electricity)
  split <mesh>          Split oversized model into parts
  quality <mesh>        Quality score (0-100)
  repair <mesh>         Fix mesh issues, ensure watertight
  predict <mesh>        Predict print failures

BATCH
  batch <dir>           Process directory of images in parallel

PRINTERS
  printers              List all supported printer profiles

MODELS
  download-model        Download TripoSR/Hunyuan3D weights
  benchmark <image>     Run performance benchmarks

MONITORING
  competitors           Show competitor tracking data
  stats                 Show local usage analytics

COMMON OPTIONS
  -o, --output          Output file path
  --format              3mf | stl | obj (default: 3mf)
  --size                Target size in mm (default: 50)
  --device              cuda | cpu (default: cuda)
  -v, --verbose         Debug logging
```

---

## API Reference

All endpoints at `http://localhost:8000`.

### Generation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/generate` | Image upload → 3D mesh file |
| POST | `/api/text-to-3d` | Text description → 3D mesh |
| POST | `/api/batch` | Multiple images → batch results |
| POST | `/api/analyze` | Image analysis without generation |

### Optimization

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/optimize` | Orientation, estimates, printability |
| POST | `/api/cost` | Cost breakdown |
| POST | `/api/split` | Split into printer-sized parts |
| POST | `/api/quality` | Quality score 0-100 |
| POST | `/api/repair` | Mesh repair → watertight |
| POST | `/api/predict` | Failure risk prediction |

### Printers

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/printers/profiles` | List all supported printer specs |
| GET | `/api/printers` | Discover Bambu printers on network |
| POST | `/api/printers/send` | Send job to Bambu printer |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/api/info` | Version, features, capabilities |
| GET | `/api/formats` | Supported formats & printer presets |
| GET | `/api/cache/stats` | Cache hit/miss stats |
| GET | `/api/stats` | Usage analytics |
| GET | `/api/benchmark/latest` | Latest benchmark report |
| GET | `/api/terms` | Terms of Service |
| WS | `/ws/progress` | Real-time pipeline progress |

---

## Supported Printers

| Printer | Build Volume | Max Speed | Auto-level |
|---------|-------------|-----------|------------|
| Bambu X1 Carbon | 256x256x256mm | 500mm/s | Yes |
| Bambu P1S | 256x256x256mm | 500mm/s | Yes |
| Bambu A1 | 256x256x256mm | 500mm/s | Yes |
| Bambu A1 Mini | 180x180x180mm | 500mm/s | Yes |
| Prusa MK4 | 250x210x220mm | 200mm/s | Yes |
| Prusa Mini+ | 180x180x180mm | 200mm/s | Yes |
| Creality Ender-3 V3 | 220x220x250mm | 250mm/s | No |
| Creality K1 | 220x220x250mm | 600mm/s | Yes |

---

## Roadmap

### Done ✅
- [x] Image/text/video to 3D pipeline with watertight guarantee
- [x] Multi-engine: TRELLIS (CVPR 2025), Tripo3D API, TripoSR, Hunyuan3D
- [x] Depth-guided mesh enhancement (single-view quality boost)
- [x] Texture mapping (projection + nearest-neighbor + UV)
- [x] Tail artifact detection & removal
- [x] CLI (20 commands), Web UI (dark/light, 3D preview, progress), REST API + SSE + WebSocket
- [x] Print optimizer, cost estimator, part splitter, quality scorer, mesh repair, failure predictor
- [x] Bambu Lab printer discovery & direct send
- [x] Async generation queue, batch processing, caching, analytics
- [x] Multi-printer profiles (8 printers), i18n (en/zh/ja), error codes
- [x] Security audit, rate limiting, content safety, API keys, billing
- [x] CI/CD (GitHub Actions), Docker support

### Next
- [ ] Tripo3D v3 model (higher quality)
- [ ] iPhone LiDAR scan → print pipeline
- [ ] On-device CoreML / MPS inference (Apple Silicon)
- [ ] Model marketplace & community templates
- [ ] Sketch to 3D (pen input)
- [ ] FEA strength analysis
- [ ] Multi-material color mapping
- [ ] Prusa/Creality direct send (OctoPrint/Klipper)
- [ ] Support structure optimizer

---

## Install

### pip (recommended)

```bash
pip install printforge
```

### From source (recommended for Mac)

```bash
git clone https://github.com/AShan0227/printforge.git
cd printforge
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Environment variables

```bash
export TRIPO_API_KEY=your_tripo_key     # Tripo3D API (300 free/month)
export HF_TOKEN=your_hf_token           # HuggingFace (for TRELLIS/TripoSR)
```

### Docker

```bash
docker compose up --build
# Web UI at http://localhost:8000
```

---

## Contributing

```bash
git clone https://github.com/printforge/printforge.git
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
