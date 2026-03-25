# Hacker News Post (Final, copy-paste ready)

---

**Title:** Show HN: PrintForge – Open-source photo to 3D print (100% watertight guarantee)

---

**Body:**

PrintForge converts a photo into a 3D-printable mesh that is guaranteed watertight. The guarantee comes from the geometry pipeline: SDF voxelization → marching cubes. The output topology is closed by construction — there is no "repair" step that might fail.

Pipeline: Photo → Hunyuan3D-2 inference (42s) → SDF voxelization → marching cubes → print optimization → 3MF/STL

Tested end-to-end on a Bambu Lab A1: 166K faces, sliced without errors (1h6m, 23.84g PLA).

Features:
- Photo-to-3D and text-to-3D
- 3MF output (Bambu Studio native) + STL
- Cost estimation (PLA/PETG/ABS/TPU/ASA/Nylon)
- Auto part-splitting with alignment pins for oversized models
- Smart orientation (minimizes support material)
- Web UI, CLI, REST API
- Docker support
- MIT licensed, runs locally

```
pip install printforge
printforge image photo.jpg -o model.3mf
```

GitHub: https://github.com/AShan0227/printforge

Tech stack: Python, Hunyuan3D-2 (via Gradio), trimesh, scikit-image (marching cubes), FastAPI.

Happy to answer questions about the watertight guarantee or the pipeline architecture.
