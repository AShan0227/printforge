# Reddit Post — r/BambuLab (Final, copy-paste ready)

**Subreddit:** r/BambuLab

---

**Title:** I made an open-source tool that converts photos to print-ready 3MF files (tested with A1, 100% watertight)

---

**Body:**

So I got tired of downloading AI-generated 3D models that look amazing in the preview but then Bambu Studio throws errors because they're full of holes and non-manifold edges. Spent way too many hours in Blender trying to fix meshes that shouldn't have been broken in the first place.

I built **PrintForge** — it takes a photo, runs it through Hunyuan3D-2 for 3D generation, then forces the output through SDF voxelization + marching cubes. The key thing: **it literally cannot produce a non-watertight mesh.** It's not "usually works" or "97% success rate" — the math guarantees it.

**Real test result on my A1:**
- Input: photo of a figurine
- AI inference: 42 seconds
- Output: 166K faces, 100% watertight
- Bambu Studio: sliced first try, 1h6m print time, 23.84g PLA
- Zero manual mesh repair needed

**How it compares to Meshy/other tools:**
- Open source (MIT) vs closed/paid
- Runs locally — your photos stay on your machine
- Free forever vs subscription
- 100% watertight vs ~97% (that 3% will ruin your day)
- Outputs native 3MF, not just STL

**Install:**
```
pip install printforge
printforge image photo.jpg -o model.3mf
```

There's also a web UI and REST API if that's more your thing.

**GitHub:** https://github.com/AShan0227/printforge

It also does cost estimation across 6 materials (PLA/PETG/ABS/TPU/ASA/Nylon), auto-splits oversized models with alignment pins, and picks the best print orientation to minimize supports.

What would you want to convert? I'm curious what people would actually use this for — I've been testing with figurines and household items but I feel like there are use cases I haven't thought of.
