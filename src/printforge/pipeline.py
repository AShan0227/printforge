"""
PrintForge Core Pipeline
========================
Image → TripoSR Inference → SDF Conversion → Marching Cubes → Watertight Mesh → Print Optimization → 3MF

Each stage is a standalone function, composable into the full pipeline.
"""

import hashlib
import io
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from .tail_remover import remove_tail

def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from env or ~/.openclaw/workspace/.hf_token file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".openclaw" / "workspace" / ".hf_token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None

import numpy as np
import requests

from .cache import ImageCache

logger = logging.getLogger(__name__)


HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/TripoSR"


@dataclass
class PipelineConfig:
    """Configuration for the PrintForge pipeline."""
    # TripoSR inference
    model_name: str = "stabilityai/TripoSR"
    device: str = "cuda"  # "cuda", "cpu", or "mps"
    inference_backend: str = "auto"  # "auto", "trellis", "hunyuan3d", "api", "local", "placeholder"

    # Background removal
    remove_background: bool = True
    foreground_ratio: float = 0.85

    # SDF / Marching Cubes
    mc_resolution: int = 256  # Marching cubes grid resolution

    # Watertight conversion
    smooth_iterations: int = 2  # Laplacian smoothing passes after MC
    adaptive_resolution: bool = True  # Scale MC resolution by model size

    # Print optimization
    min_wall_thickness_mm: float = 0.4  # FDM minimum
    max_faces: int = 200_000  # Simplify if exceeding
    add_base: bool = False  # Add flat base for easier printing
    base_height_mm: float = 2.0

    # Multi-view
    multi_view: bool = False  # Generate multiple views for better reconstruction
    multi_view_count: int = 4  # front/back/left/right

    # Output
    output_format: str = "3mf"  # "3mf" or "stl"
    scale_mm: float = 50.0  # Default size in mm


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""
    mesh_path: str
    vertices: int
    faces: int
    is_watertight: bool
    wall_thickness_ok: bool
    duration_ms: float
    stages: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)


class PrintForgePipeline:
    """Main pipeline: Image → 3D printable mesh."""
    
    def __init__(self, config: Optional[PipelineConfig] = None, cache: Optional[ImageCache] = None):
        self.config = config or PipelineConfig()
        self._model = None
        self._cache = cache or ImageCache()
    
    def run(self, image_path: str, output_path: str, progress_callback=None) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            progress_callback: Optional callable(stage: str, progress: float)
                               for real-time progress updates.
        """
        def _progress(stage: str, pct: float):
            if progress_callback:
                progress_callback(stage, pct)

        start = time.time()
        stages = {}
        warnings = []

        # Cache check: compute SHA256 of input image and look up cached result
        image_bytes = Path(image_path).read_bytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        cached_path = self._cache.get(image_bytes)

        if cached_path and Path(cached_path).exists():
            import trimesh
            import shutil

            logger.info("Cache hit for image %s — loading cached mesh", image_hash[:12])
            _progress("cache_hit", 0.0)
            cached_mesh = trimesh.load(cached_path, force="mesh")

            # Copy cached file to output_path
            shutil.copy2(cached_path, output_path)

            total_ms = (time.time() - start) * 1000
            _progress("done", 1.0)
            return PipelineResult(
                mesh_path=output_path,
                vertices=len(cached_mesh.vertices),
                faces=len(cached_mesh.faces),
                is_watertight=bool(cached_mesh.is_watertight),
                wall_thickness_ok=True,
                duration_ms=total_ms,
                stages={"cache_hit": total_ms / 1000},
                warnings=[],
            )

        # Stage 1: Load and preprocess image
        logger.info("Stage 1: Loading image...")
        _progress("load_image", 0.0)
        t0 = time.time()
        image = self._load_image(image_path)
        stages["load_image"] = time.time() - t0

        # Stage 1.5: Background removal
        if self.config.remove_background:
            logger.info("Stage 1.5: Removing background...")
            _progress("remove_background", 0.15)
            t0 = time.time()
            image = self._remove_background(image)
            stages["remove_background"] = time.time() - t0

        # Stage 2: 3D inference (single or multi-view)
        if self.config.multi_view:
            logger.info("Stage 2: Multi-view inference...")
            _progress("multi_view", 0.20)
            t0 = time.time()
            raw_mesh = self._infer_multi_view(image)
            stages["multi_view_inference"] = time.time() - t0
        else:
            logger.info("Stage 2: TripoSR inference...")
            _progress("inference", 0.25)
            t0 = time.time()
            raw_mesh = self._infer_3d(image)
            stages["inference"] = time.time() - t0
        
        # Stage 3: SDF conversion → watertight mesh
        logger.info("Stage 3: SDF watertight conversion...")
        _progress("watertight", 0.50)
        t0 = time.time()
        watertight_mesh = self._make_watertight(raw_mesh)
        stages["watertight"] = time.time() - t0

        # Stage 3.5: Tail detection & removal
        logger.info("Stage 3.5: Tail detection & removal...")
        _progress("tail_removal", 0.62)
        t0 = time.time()
        tail_result = remove_tail(watertight_mesh)
        stages["tail_removal"] = time.time() - t0
        if tail_result.tail_detected:
            tail_warn = (
                f"Tail detected along {tail_result.tail_direction}: "
                f"{tail_result.removed_percentage*100:.1f}% vertices removed"
            )
            logger.warning(tail_warn)
            warnings.append(tail_warn)
            watertight_mesh = trimesh.Trimesh(
                vertices=tail_result.cleaned_verts,
                faces=watertight_mesh.faces,
            )

        # Stage 4: Print optimization
        logger.info("Stage 4: Print optimization...")
        _progress("optimization", 0.70)
        t0 = time.time()
        optimized_mesh, opt_warnings = self._optimize_for_print(watertight_mesh)
        warnings.extend(opt_warnings)
        stages["optimization"] = time.time() - t0

        # Stage 5: Export
        logger.info("Stage 5: Exporting...")
        _progress("export", 0.90)
        t0 = time.time()
        self._export(optimized_mesh, output_path)
        stages["export"] = time.time() - t0
        
        # Cache the result for future lookups
        try:
            self._cache.put(image_bytes, output_path)
        except Exception as e:
            logger.warning("Failed to cache result: %s", e)

        total_ms = (time.time() - start) * 1000

        result = PipelineResult(
            mesh_path=output_path,
            vertices=len(optimized_mesh.vertices) if optimized_mesh else 0,
            faces=len(optimized_mesh.faces) if optimized_mesh else 0,
            is_watertight=optimized_mesh.is_watertight if optimized_mesh else False,
            wall_thickness_ok=len([w for w in warnings if "wall" in w.lower()]) == 0,
            duration_ms=total_ms,
            stages=stages,
            warnings=warnings,
        )
        
        _progress("done", 1.0)
        logger.info(f"Pipeline complete: {result.vertices} vertices, {result.faces} faces, "
                     f"watertight={result.is_watertight}, {total_ms:.0f}ms")
        return result
    
    def _load_image(self, image_path: str):
        """Load and preprocess image for TripoSR."""
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        # TripoSR expects 512x512
        img = img.resize((512, 512), Image.LANCZOS)
        return img

    def _remove_background(self, image):
        """Remove background using rembg for cleaner TripoSR inference."""
        try:
            from rembg import remove as rembg_remove
        except ImportError:
            logger.warning("rembg not installed — skipping background removal")
            return image

        # rembg returns RGBA
        result = rembg_remove(image)
        # Composite onto neutral gray background (TripoSR convention)
        result_np = np.array(result).astype(np.float32) / 255.0
        rgb = result_np[:, :, :3]
        alpha = result_np[:, :, 3:4]
        composited = rgb * alpha + (1 - alpha) * 0.5

        from PIL import Image as PILImage
        # Resize foreground to fill ~85% of frame
        composited_img = PILImage.fromarray(
            (composited * 255.0).astype(np.uint8)
        ).convert("RGB")
        return composited_img

    def _infer_3d(self, image):
        """Run 3D inference using the configured backend with a full fallback chain.

        Fallback order (when backend is 'auto'):
            1. Hunyuan3D-2 (full model via Gradio)
            2. Hunyuan3D-2mini (lighter model via Gradio)
            3. TripoSR via HuggingFace Inference API
            4. Local TripoSR inference
            5. Placeholder mesh (last resort)
        """
        backend = self.config.inference_backend

        if backend == "placeholder":
            logger.info("Using placeholder mesh (configured)")
            return self._create_placeholder_mesh()

        # Step 0: TRELLIS (best quality, CVPR 2025 — preferred over Hunyuan3D)
        if backend in ("auto", "trellis"):
            mesh = self._infer_trellis(image)
            if mesh is not None:
                return mesh
            if backend == "trellis":
                raise RuntimeError("TRELLIS inference failed and backend is 'trellis'")
            logger.warning("Fallback: TRELLIS failed, trying Hunyuan3D-2...")

        # Step 1: Hunyuan3D-2 (full)
        if backend in ("auto", "hunyuan3d"):
            mesh = self._infer_hunyuan3d(image)
            if mesh is not None:
                return mesh
            if backend == "hunyuan3d":
                raise RuntimeError("Hunyuan3D inference failed and backend is 'hunyuan3d'")
            logger.warning("Fallback: Hunyuan3D-2 failed, trying Hunyuan3D-2mini...")

        # Step 2: Hunyuan3D-2mini (lighter)
        if backend in ("auto",):
            mesh = self._infer_hunyuan3d_mini(image)
            if mesh is not None:
                return mesh
            logger.warning("Fallback: Hunyuan3D-2mini failed, trying TripoSR HF API...")

        # Step 3: TripoSR via HuggingFace API
        if backend in ("auto", "api"):
            mesh = self._infer_hf_api(image)
            if mesh is not None:
                return mesh
            if backend == "api":
                raise RuntimeError("HuggingFace API inference failed and backend is 'api'")
            logger.warning("Fallback: HF API failed, trying local inference...")

        # Step 4: Local TripoSR
        if backend in ("auto", "local"):
            mesh = self._infer_local(image)
            if mesh is not None:
                return mesh
            if backend == "local":
                raise RuntimeError("Local TripoSR inference failed and backend is 'local'")
            logger.warning("Fallback: Local inference failed, using placeholder mesh...")

        # Step 5: Placeholder (last resort)
        logger.warning("All inference backends unavailable — using placeholder mesh")
        return self._create_placeholder_mesh()

    def _infer_trellis(self, image):
        """Run inference via TRELLIS (CVPR 2025) — best single-image-to-3D quality.

        TRELLIS uses Structured Latent Representation (SLAT) with 2B parameters.
        Unlike Hunyuan3D, it doesn't need fake multi-view synthesis, so it avoids
        the "tail" artifact caused by contradictory synthetic views.
        """
        try:
            from .trellis_backend import TrellisBackend
        except ImportError:
            logger.info("trellis_backend not available — skipping TRELLIS")
            return None

        try:
            backend = TrellisBackend()
            result = backend.generate(image)
            if result is not None:
                logger.info(
                    f"TRELLIS inference OK: {result.vertices} verts, "
                    f"{result.faces} faces"
                )
                return result.mesh
            return None
        except Exception as e:
            logger.warning(f"TRELLIS inference failed: {e}")
            return None

    def _infer_hunyuan3d(self, image):
        """Run inference via Hunyuan3D-2 Gradio Space with multi-view enhancement.

        Hunyuan3D-2's /shape_generation endpoint accepts:
            (text_prompt, front_image, back_image, left_image, ...)
        We synthesize back/left views from the front image for better results.
        """
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            logger.info("gradio_client not installed — skipping Hunyuan3D backend")
            return None

        try:
            import trimesh
            import tempfile
            from .multi_view import MultiViewEnhancer

            # Generate multi-view approximations from the front image
            enhancer = MultiViewEnhancer()
            views = enhancer.enhance_from_pil(image)

            temp_files = []

            def _save_temp(pil_img):
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                pil_img.save(tmp, format="PNG")
                tmp.close()
                temp_files.append(tmp.name)
                return tmp.name

            front_path = _save_temp(views["front"])
            back_path = _save_temp(views["back"])
            left_path = _save_temp(views["left"])

            hf_token = _get_hf_token()
            if hf_token:
                logger.info("Using authenticated HF access (increased ZeroGPU quota)")
                client = Client("Tencent/Hunyuan3D-2", token=hf_token)
            else:
                logger.info("No HF_TOKEN set — using anonymous access (limited quota)")
                client = Client("Tencent/Hunyuan3D-2")
            result = client.predict(
                "",
                handle_file(front_path),
                handle_file(back_path),
                handle_file(left_path),
                None,
                api_name="/shape_generation",
            )

            # result[0]['value'] contains the GLB file path
            glb_path = result[0]["value"]
            mesh = trimesh.load(glb_path, file_type="glb", force="mesh")
            logger.info(f"Hunyuan3D inference OK: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

            # Clean up temp files
            for f in temp_files:
                try:
                    os.unlink(f)
                except OSError:
                    pass
            return mesh
        except Exception as e:
            logger.warning(f"Hunyuan3D inference failed: {e}")
            return None

    def _infer_hunyuan3d_mini(self, image):
        """Run inference via Hunyuan3D-2mini Gradio Space (lighter model)."""
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            logger.info("gradio_client not installed — skipping Hunyuan3D-2mini backend")
            return None

        try:
            import trimesh
            import tempfile

            # Save PIL image to temp file for handle_file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp, format="PNG")
                temp_path = tmp.name

            hf_token = _get_hf_token()
            if hf_token:
                logger.info("Using authenticated HF access for Hunyuan3D-2mini")
                client = Client("Tencent/Hunyuan3D-2mini", token=hf_token)
            else:
                logger.info("No HF_TOKEN set — using anonymous access for Hunyuan3D-2mini")
                client = Client("Tencent/Hunyuan3D-2mini")
            result = client.predict(
                "", handle_file(temp_path), None, None, None,
                api_name="/shape_generation",
            )

            # result[0]['value'] contains the GLB file path
            glb_path = result[0]["value"]
            mesh = trimesh.load(glb_path, file_type="glb", force="mesh")
            logger.info(f"Hunyuan3D-2mini inference OK: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

            # Clean up temp file
            os.unlink(temp_path)
            return mesh
        except Exception as e:
            logger.warning(f"Hunyuan3D-2mini inference failed: {e}")
            return None

    def _infer_hf_api(self, image):
        """Run inference via HuggingFace Inference API."""
        hf_token = _get_hf_token()
        if not hf_token:
            logger.info("HF_TOKEN not set — skipping HuggingFace API backend")
            return None

        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            resp = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {hf_token}"},
                data=image_bytes,
                timeout=120,
            )
            resp.raise_for_status()

            # Response is GLB binary — load via trimesh
            import trimesh
            glb_data = io.BytesIO(resp.content)
            mesh = trimesh.load(glb_data, file_type="glb", force="mesh")
            logger.info(f"HF API inference OK: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
            return mesh
        except Exception as e:
            logger.warning(f"HuggingFace API inference failed: {e}")
            return None

    def _infer_local(self, image):
        """Run inference using local TripoSR model.

        Supports both old API (scene.get_mesh) and new API (model.extract_mesh).
        """
        try:
            from tsr.system import TSR
            import torch
        except ImportError:
            logger.info("tsr package not available — skipping local backend")
            return None

        try:
            if self._model is None:
                # Pick best available device
                device = self.config.device
                if device == "cuda" and not torch.cuda.is_available():
                    device = "mps" if torch.backends.mps.is_available() else "cpu"
                elif device == "mps" and not torch.backends.mps.is_available():
                    device = "cpu"
                self.config.device = device

                logger.info(f"Loading TripoSR model: {self.config.model_name} on {device}")
                self._model = TSR.from_pretrained(
                    self.config.model_name,
                    config_name="config.yaml",
                    weight_name="model.ckpt",
                )
                self._model.to(device)

            with torch.no_grad():
                scene_codes = self._model(image, device=self.config.device)

            # New API: model returns scene_codes tensor, use extract_mesh
            if hasattr(scene_codes, 'shape') and hasattr(self._model, 'extract_mesh'):
                meshes = self._model.extract_mesh(
                    scene_codes, 
                    resolution=self.config.mc_resolution,
                    has_vertex_color=False,
                )
                raw = meshes[0]
                verts = raw.vertices.cpu().numpy() if hasattr(raw.vertices, 'cpu') else raw.vertices
                faces = raw.faces.cpu().numpy() if hasattr(raw.faces, 'cpu') else raw.faces
                import trimesh
                mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            else:
                # Old API: scene object with get_mesh
                mesh = scene_codes.get_mesh(resolution=self.config.mc_resolution)

            logger.info(f"Local inference OK: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
            return mesh
        except Exception as e:
            logger.warning(f"Local TripoSR inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _infer_multi_view(self, image):
        """Run multi-view inference: generate views → infer each → merge meshes.

        Uses the MultiViewEnhancer to create front/back/left/right views,
        runs TripoSR on each view, then merges the resulting meshes.
        """
        import trimesh
        from .multi_view import MultiViewEnhancer

        enhancer = MultiViewEnhancer()
        views = enhancer.enhance_from_pil(image)
        logger.info(f"Multi-view: generated {len(views)} views")

        meshes = []
        for view_name, view_img in views.items():
            logger.info(f"Multi-view: inferring {view_name}...")
            mesh = self._infer_3d(view_img)
            if mesh is not None:
                if not isinstance(mesh, trimesh.Trimesh):
                    if hasattr(mesh, 'vertices') and hasattr(mesh, 'faces'):
                        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
                meshes.append(mesh)
                logger.info(f"  {view_name}: {len(mesh.vertices)} verts")
            else:
                logger.warning(f"  {view_name}: inference failed, skipping")

        if not meshes:
            logger.error("Multi-view: all views failed, falling back to placeholder")
            return self._create_placeholder_mesh()

        if len(meshes) == 1:
            return meshes[0]

        # Merge strategy: use the front view as primary (best quality),
        # average vertex positions with overlapping meshes for better coverage
        # Simple approach: concatenate all meshes (more vertices = more detail)
        merged = trimesh.util.concatenate(meshes)
        logger.info(f"Multi-view: merged {len(meshes)} meshes → {len(merged.vertices)} verts, {len(merged.faces)} faces")
        return merged

    def _create_placeholder_mesh(self):
        """Create a simple cube mesh for testing without TripoSR."""
        import trimesh
        return trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    
    def _choose_mc_resolution(self, mesh) -> int:
        """Pick marching-cubes resolution based on model bounding-box size.

        Small objects (< 30mm):  96
        Medium (30-100mm):      128
        Large (> 100mm):        192

        Falls back to self.config.mc_resolution when adaptive_resolution is False.
        """
        if not self.config.adaptive_resolution:
            return self.config.mc_resolution

        extent = mesh.bounding_box.extents.max()
        if extent < 30:
            return 96
        elif extent <= 100:
            return 128
        else:
            return 192

    def _make_watertight(self, mesh):
        """Convert mesh to watertight using SDF → Marching Cubes.

        Applies adaptive MC resolution and optional Laplacian smoothing.
        """
        import trimesh

        if hasattr(mesh, "is_watertight") and mesh.is_watertight:
            logger.info("Mesh is already watertight, skipping SDF conversion")
            return mesh

        # Convert to trimesh if not already
        if not isinstance(mesh, trimesh.Trimesh):
            if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                mesh = trimesh.Trimesh(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                )
            else:
                raise ValueError("Cannot convert mesh to trimesh format")

        # Method: Voxelize → Fill → Marching Cubes → guaranteed watertight
        try:
            resolution = self._choose_mc_resolution(mesh)
            pitch = mesh.bounding_box.extents.max() / resolution
            voxel_grid = mesh.voxelized(pitch).fill()

            # Marching cubes on filled voxels → guaranteed watertight
            watertight = voxel_grid.marching_cubes

            # Laplacian smoothing preserves shape while removing staircase artifacts
            if self.config.smooth_iterations > 0:
                trimesh.smoothing.filter_laplacian(
                    watertight, iterations=self.config.smooth_iterations
                )

            logger.info(f"SDF conversion: {len(mesh.faces)} → {len(watertight.faces)} faces, "
                        f"resolution={resolution}, smooth={self.config.smooth_iterations}, "
                        f"watertight={watertight.is_watertight}")
            return watertight

        except Exception as e:
            logger.warning(f"SDF conversion failed ({e}), attempting repair instead")
            # Fallback: use trimesh repair
            trimesh.repair.fix_normals(mesh)
            trimesh.repair.fix_winding(mesh)
            trimesh.repair.fill_holes(mesh)
            return mesh
    
    def _optimize_for_print(self, mesh):
        """Optimize mesh for 3D printing."""
        import trimesh
        warnings = []
        
        # 1. Scale to target size
        current_size = mesh.bounding_box.extents.max()
        if current_size > 0:
            scale_factor = self.config.scale_mm / current_size
            mesh.apply_scale(scale_factor)
        
        # 2. Simplify if too many faces (numpy subsampling — no native deps)
        if len(mesh.faces) > self.config.max_faces:
            original_count = len(mesh.faces)
            step = max(1, len(mesh.faces) // self.config.max_faces)
            keep = np.arange(0, len(mesh.faces), step)[:self.config.max_faces]
            mesh = mesh.submesh([keep], append=True)
            logger.info(f"Simplified: {original_count} → {len(mesh.faces)} faces")
        
        # 3. Fix normals
        trimesh.repair.fix_normals(mesh)
        
        # 4. Check wall thickness (approximate)
        if hasattr(mesh, "bounding_box"):
            min_extent = mesh.bounding_box.extents.min()
            if min_extent < self.config.min_wall_thickness_mm:
                warnings.append(f"Wall thickness warning: min extent {min_extent:.2f}mm < {self.config.min_wall_thickness_mm}mm")
        
        # 5. Add base if requested
        if self.config.add_base:
            base = trimesh.creation.box(
                extents=[
                    mesh.bounding_box.extents[0] * 1.1,
                    mesh.bounding_box.extents[1] * 1.1,
                    self.config.base_height_mm,
                ]
            )
            base.apply_translation([
                mesh.bounding_box.centroid[0],
                mesh.bounding_box.centroid[1],
                mesh.bounds[0][2] - self.config.base_height_mm / 2,
            ])
            mesh = trimesh.util.concatenate([mesh, base])
        
        # 6. Final watertight check
        if not mesh.is_watertight:
            warnings.append("Final mesh is not watertight. May cause slicing issues.")
            trimesh.repair.fill_holes(mesh)
        
        return mesh, warnings
    
    def _export(self, mesh, output_path: str):
        """Export mesh in the configured format."""
        output_path = Path(output_path)

        if self.config.output_format == "3mf" or output_path.suffix == ".3mf":
            # 3MF export
            try:
                mesh.export(str(output_path), file_type="3mf")
            except Exception:
                stl_path = output_path.with_suffix(".stl")
                mesh.export(str(stl_path), file_type="stl")
                logger.warning(f"3MF export failed, saved as STL: {stl_path}")
        elif self.config.output_format == "glb" or output_path.suffix == ".glb":
            from .export_glb import export_glb
            export_glb(mesh, str(output_path))
        else:
            mesh.export(str(output_path), file_type="stl")
