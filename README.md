# 🏭 PrintForge

**一张照片，一键 3D 打印。**

基于 TripoSR + SDF Watertight Pipeline，专为拓竹等 FDM 打印机优化。

## Quick Start

```bash
pip install -e .
printforge photo.jpg -o model.3mf
```

## Architecture

```
Image → TripoSR (DINOv1 → Triplane → NeRF) → SDF → Marching Cubes → Watertight Mesh → Print Optimization → 3MF
```

## License

MIT
