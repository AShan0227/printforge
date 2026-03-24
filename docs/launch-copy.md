# PrintForge Launch Copy

## Show HN Post

**Title:** Show HN: PrintForge – Open-source photo to 3D print (watertight guaranteed)

**Body:**

I built PrintForge because I kept downloading 3D models that looked great but failed to slice.

PrintForge takes a photo, generates a 3D model using Hunyuan3D-2, then guarantees a watertight mesh through SDF voxelization + marching cubes. The output is a 3MF file that opens directly in Bambu Studio — tested with a real Bambu Lab A1 printer.

Key features:
- 📷 Photo → 3D model in 42 seconds
- 🔒 100% watertight guarantee (not "usually works")
- 🖨️ Outputs 3MF (Bambu) + STL
- 💰 Cost estimator (6 materials: PLA/PETG/ABS/TPU/ASA/Nylon)
- ✂️ Auto-splits oversized models with alignment pins
- 📐 Smart orientation (minimizes support material)
- 🌐 Web UI + CLI + REST API
- 🐳 Docker support
- MIT licensed, runs locally (your images stay on your machine)

```bash
pip install printforge
printforge image photo.jpg -o model.3mf
```

GitHub: https://github.com/AShan0227/printforge

Tech: Python + TripoSR/Hunyuan3D + trimesh + scikit-image (marching cubes) + FastAPI

I'd love feedback, especially from anyone who's tried printing AI-generated models.

---

## Reddit r/BambuLab Post

**Title:** I made an open-source tool that converts photos to print-ready 3MF files (tested with A1)

**Body:**

Hey everyone! I built PrintForge — an open-source tool that takes a photo and produces a watertight 3MF file you can directly import into Bambu Studio.

**The problem:** AI 3D generators make cool models, but most aren't print-ready (holes, non-manifold edges, etc.)

**The solution:** PrintForge forces watertight output through SDF voxelization + marching cubes. It literally cannot produce a non-watertight mesh.

Just tested with my A1: Photo → 42s AI inference → watertight mesh → Bambu Studio → sliced successfully (1h6m, 23.84g PLA).

Features: Photo to 3D, Text to 3D, auto part splitting, cost estimation, smart orientation.

GitHub: https://github.com/AShan0227/printforge

MIT licensed, runs locally. No cloud upload needed.

What models would you want to try converting?

---

## Product Hunt

**Tagline:** One photo to 3D print. Open-source, watertight guaranteed.

**Description:**
PrintForge converts any photo into a 3D-printable file in 42 seconds. Unlike other tools, it guarantees watertight output through SDF voxelization — your model will always slice successfully. MIT licensed, runs locally, outputs Bambu Studio-compatible 3MF.

**Topics:** 3D Printing, Open Source, Artificial Intelligence, Design Tools, Maker
