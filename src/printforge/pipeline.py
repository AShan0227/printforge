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
TRIPO_API_BASE = "https://api.tripo3d.ai/v2/openapi"


@dataclass
class PipelineConfig:
    """Configuration for the PrintForge pipeline."""
    # TripoSR inference
    model_name: str = "stabilityai/TripoSR"
    device: str = "cuda"  # "cuda", "cpu", or "mps"
    inference_backend: str = "auto"  # "auto", "trellis", "tripo", "hunyuan3d", "api", "local", "placeholder"

    # Auto-crop to subject
    auto_crop: bool = True

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

    # Depth enhancement
    use_depth: bool = True  # Use depth estimation to enhance 3D reconstruction
    depth_weight: float = 0.3  # How much depth map influences vertex displacement (0-1)
    depth_model: str = ""  # HF model for depth; empty = auto

    # Texture
    apply_texture: bool = False  # Apply image colors to mesh
    texture_method: str = "projection"  # "projection" or "nearest"

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

        # Stage 1.2: Auto-crop to subject
        if self.config.auto_crop:
            logger.info("Stage 1.2: Auto-cropping to subject...")
            _progress("auto_crop", 0.08)
            t0 = time.time()
            try:
                from .auto_crop import auto_crop
                original_size = image.size
                image = auto_crop(image, padding=0.12)
                if image.size != original_size:
                    logger.info(f"Auto-cropped: {original_size} → {image.size}")
                stages["auto_crop"] = time.time() - t0
            except Exception as e:
                logger.warning(f"Auto-crop failed (non-fatal): {e}")
                stages["auto_crop"] = time.time() - t0

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
        
        # Stage 3: Watertight conversion
        # High-quality API outputs (Tripo, TRELLIS) should NOT be voxel-rebuilt
        # — it destroys geometry detail. Only do light repair.
        is_api_backend = self.config.inference_backend in ("tripo", "trellis", "auto")
        is_high_quality = raw_mesh is not None and len(raw_mesh.vertices) > 5000

        if is_api_backend and is_high_quality:
            logger.info("Stage 3: Light mesh repair (preserving API geometry)...")
            _progress("watertight", 0.50)
            t0 = time.time()
            watertight_mesh = self._light_repair(raw_mesh)
            stages["watertight"] = time.time() - t0
        else:
            logger.info("Stage 3: SDF watertight conversion...")
            _progress("watertight", 0.50)
            t0 = time.time()
            watertight_mesh = self._make_watertight(raw_mesh)
            stages["watertight"] = time.time() - t0

        # Stage 3.5: Tail detection & removal (skip for high-quality API output)
        if not (is_api_backend and is_high_quality):
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
        else:
            logger.info("Stage 3.5: Skipping tail removal (high-quality API output)")
            _progress("tail_removal", 0.62)
            stages["tail_removal"] = 0.0

        # Stage 3.7: Depth-guided enhancement (skip for high-quality API output)
        if self.config.use_depth and not (is_api_backend and is_high_quality):
            logger.info("Stage 3.7: Depth-guided mesh enhancement...")
            _progress("depth_enhance", 0.65)
            t0 = time.time()
            try:
                watertight_mesh, depth_warns = self._apply_depth_enhancement(watertight_mesh, image)
                warnings.extend(depth_warns)
                stages["depth_enhance"] = time.time() - t0
            except Exception as e:
                logger.warning(f"Depth enhancement failed (non-fatal): {e}")
                stages["depth_enhance"] = time.time() - t0

        # Stage 4: Print optimization
        logger.info("Stage 4: Print optimization...")
        _progress("optimization", 0.70)
        t0 = time.time()
        optimized_mesh, opt_warnings = self._optimize_for_print(watertight_mesh)
        warnings.extend(opt_warnings)
        stages["optimization"] = time.time() - t0

        # Stage 4.5: Texture (optional)
        if self.config.apply_texture or (
            self.config.output_format == "glb"
            and Path(output_path).suffix.lower() == ".glb"
        ):
            logger.info("Stage 4.5: Applying texture...")
            _progress("texture", 0.85)
            t0 = time.time()
            try:
                from .texture import TextureMapper
                mapper = TextureMapper()
                optimized_mesh = mapper.apply_vertex_colors(
                    optimized_mesh, image, method=self.config.texture_method
                )
                stages["texture"] = time.time() - t0
            except Exception as e:
                logger.warning(f"Texture application failed: {e}")
                stages["texture"] = time.time() - t0

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

        # Step 3: Tripo API — try multiview first if reference images available
        if backend in ("auto", "tripo"):
            # Try multiview with synthesized views (better quality)
            if self.config.multi_view:
                mesh = self._infer_tripo_multiview(image)
                if mesh is not None:
                    return mesh
                logger.warning("Multiview failed, falling back to single image...")

            mesh = self._infer_tripo_api(image)
            if mesh is not None:
                return mesh
            if backend == "tripo":
                raise RuntimeError("Tripo API inference failed and backend is 'tripo'")
            logger.warning("Fallback: Tripo API failed, trying HF API...")

        # Step 3.5: TripoSR via HuggingFace API
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

    def _infer_tripo_multiview(self, image):
        """Run Tripo multiview_to_model with synthesized multi-angle views.

        Uses Zero123++ or simple view synthesis to generate left/back/right views,
        then submits all views to Tripo for higher-quality 3D reconstruction.
        """
        tripo_key = os.environ.get("TRIPO_API_KEY")
        if not tripo_key:
            return None

        try:
            import trimesh
            from .multi_view import MultiViewEnhancer

            headers = {"Authorization": f"Bearer {tripo_key}"}

            # Step 1: Generate multi-angle views
            logger.info("Tripo multiview: synthesizing views...")
            enhancer = MultiViewEnhancer()
            views = enhancer.enhance_from_pil(image)

            # Upload all views
            def upload(pil_img, name):
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                buf.seek(0)
                r = requests.post(
                    f"{TRIPO_API_BASE}/upload",
                    headers=headers,
                    files={"file": (name, buf, "image/png")},
                    timeout=15,
                )
                r.raise_for_status()
                return r.json()["data"]["image_token"]

            # Upload front (original) + synthesized views
            front_buf = io.BytesIO()
            image.save(front_buf, format="PNG")
            front_buf.seek(0)
            front_token = requests.post(
                f"{TRIPO_API_BASE}/upload", headers=headers,
                files={"file": ("front.png", front_buf, "image/png")}, timeout=15,
            ).json()["data"]["image_token"]

            tokens = {"front": front_token}
            view_map = {"left": "left", "back": "back", "right": "right"}
            for view_name, key in view_map.items():
                if key in views:
                    tokens[view_name] = upload(views[key], f"{view_name}.png")

            logger.info(f"Tripo multiview: uploaded {len(tokens)} views")

            # Step 2: Create multiview task
            task_files = [{"type": "png", "file_token": tokens["front"]}]
            for vn in ["left", "back", "right"]:
                if vn in tokens:
                    task_files.append({"type": "png", "file_token": tokens[vn]})

            task_resp = requests.post(
                f"{TRIPO_API_BASE}/task",
                headers={**headers, "Content-Type": "application/json"},
                json={"type": "multiview_to_model", "files": task_files},
                timeout=15,
            )
            task_resp.raise_for_status()
            task_data = task_resp.json()
            if task_data.get("code") != 0:
                logger.warning(f"Tripo multiview task failed: {task_data}")
                return None

            task_id = task_data["data"]["task_id"]
            logger.info(f"Tripo multiview: task {task_id}, polling...")

            # Step 3: Poll
            import time as _time
            for _ in range(60):
                _time.sleep(3)
                poll = requests.get(f"{TRIPO_API_BASE}/task/{task_id}", headers=headers, timeout=10)
                status = poll.json().get("data", {}).get("status", "?")
                if status == "success":
                    break
                elif status in ("failed", "cancelled"):
                    logger.warning(f"Tripo multiview failed: {status}")
                    return None

            # Step 4: Download
            output = poll.json().get("data", {}).get("output", {})
            result_data = poll.json().get("data", {}).get("result", {})
            model_url = None
            for source in [output, result_data]:
                for key in ["pbr_model", "model"]:
                    val = source.get(key)
                    if isinstance(val, str) and val.startswith("http"):
                        model_url = val; break
                    elif isinstance(val, dict) and val.get("url"):
                        model_url = val["url"]; break
                if model_url: break

            if not model_url:
                return None

            glb = requests.get(model_url, timeout=60)
            mesh = trimesh.load(io.BytesIO(glb.content), file_type="glb", force="mesh")
            logger.info(f"Tripo multiview OK: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
            return mesh

        except Exception as e:
            logger.warning(f"Tripo multiview failed: {e}")
            return None

    def _infer_tripo_api(self, image):
        """Run inference via Tripo3D API (image → 3D model).

        Tripo offers 300 free credits/month with high-quality output.
        Flow: upload image → create task → poll → download GLB.
        """
        tripo_key = os.environ.get("TRIPO_API_KEY")
        if not tripo_key:
            logger.info("TRIPO_API_KEY not set — skipping Tripo backend")
            return None

        try:
            import trimesh

            headers = {"Authorization": f"Bearer {tripo_key}"}

            # Step 1: Upload image
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)

            upload_resp = requests.post(
                f"{TRIPO_API_BASE}/upload",
                headers=headers,
                files={"file": ("input.png", buf, "image/png")},
                timeout=30,
            )
            upload_resp.raise_for_status()
            upload_data = upload_resp.json()

            if upload_data.get("code") != 0:
                logger.warning(f"Tripo upload failed: {upload_data}")
                return None

            file_token = upload_data["data"]["image_token"]
            logger.info(f"Tripo: image uploaded, token={file_token[:20]}...")

            # Step 2: Create task
            task_resp = requests.post(
                f"{TRIPO_API_BASE}/task",
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "type": "image_to_model",
                    "file": {"type": "png", "file_token": file_token},
                },
                timeout=30,
            )
            task_resp.raise_for_status()
            task_data = task_resp.json()

            if task_data.get("code") != 0:
                logger.warning(f"Tripo task creation failed: {task_data}")
                return None

            task_id = task_data["data"]["task_id"]
            logger.info(f"Tripo: task created {task_id}, polling...")

            # Step 3: Poll for completion (max 120s)
            import time as _time
            for _ in range(60):
                _time.sleep(2)
                poll_resp = requests.get(
                    f"{TRIPO_API_BASE}/task/{task_id}",
                    headers=headers,
                    timeout=15,
                )
                poll_resp.raise_for_status()
                poll_data = poll_resp.json()

                status = poll_data.get("data", {}).get("status", "unknown")
                if status == "success":
                    break
                elif status in ("failed", "cancelled"):
                    logger.warning(f"Tripo task {status}: {poll_data}")
                    return None
                # else: queued, running — keep polling
            else:
                logger.warning("Tripo: task timed out after 120s")
                return None

            # Step 4: Get model URL and download
            output = poll_data.get("data", {}).get("output", {})
            result_data = poll_data.get("data", {}).get("result", {})

            model_url = None
            # Try multiple response formats
            for source in [output, result_data]:
                if not source:
                    continue
                for key in ["pbr_model", "model", "base_model"]:
                    val = source.get(key)
                    if isinstance(val, str) and val.startswith("http"):
                        model_url = val
                        break
                    elif isinstance(val, dict) and val.get("url"):
                        model_url = val["url"]
                        break
                if model_url:
                    break

            if not model_url:
                logger.warning(f"Tripo: no model URL in response: {poll_data}")
                return None

            logger.info(f"Tripo: downloading GLB from {model_url[:60]}...")
            glb_resp = requests.get(model_url, timeout=60)
            glb_resp.raise_for_status()

            # Load preserving texture — try Scene first, extract mesh with visual
            loaded = trimesh.load(io.BytesIO(glb_resp.content), file_type="glb")
            if isinstance(loaded, trimesh.Scene):
                meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if meshes:
                    mesh = meshes[0]  # Primary mesh with texture
                    if len(meshes) > 1:
                        # Concatenate but keep first mesh's visual
                        visual = mesh.visual
                        mesh = trimesh.util.concatenate(meshes)
                        mesh.visual = visual
                else:
                    mesh = trimesh.load(io.BytesIO(glb_resp.content), file_type="glb", force="mesh")
            else:
                mesh = loaded

            logger.info(
                f"Tripo API OK: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
                f"visual={mesh.visual.kind if hasattr(mesh.visual, 'kind') else 'none'}"
            )
            return mesh

        except Exception as e:
            logger.warning(f"Tripo API inference failed: {e}")
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
    
    def _apply_depth_enhancement(self, mesh, image):
        """Enhance mesh geometry using depth estimation.

        Uses estimated depth map to displace mesh vertices along their normals,
        adding geometric detail that single-view 3D reconstruction misses
        (especially for the back/sides of objects).

        Args:
            mesh: trimesh.Trimesh — the watertight mesh
            image: PIL.Image — original input image

        Returns:
            (enhanced_mesh, warnings)
        """
        import trimesh
        from .depth_estimator import DepthEstimator

        warnings = []

        try:
            estimator = DepthEstimator(model=self.config.depth_model or None)
            depth_result = estimator.estimate(image)
        except Exception as e:
            warnings.append(f"Depth estimation failed: {e}")
            return mesh, warnings

        depth_map = depth_result.depth_map
        depth_ratio = depth_result.estimated_depth_ratio
        weight = self.config.depth_weight

        logger.info(
            f"Depth enhancement: ratio={depth_ratio:.2f}, weight={weight}, "
            f"map shape={depth_map.shape}"
        )

        # Only enhance if depth ratio suggests the object has meaningful depth
        if depth_ratio < 0.05:
            warnings.append("Depth ratio too low — object appears flat, skipping enhancement")
            return mesh, warnings

        verts = mesh.vertices.copy()

        # Compute vertex normals
        if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
            mesh.fix_normals()
        normals = mesh.vertex_normals

        # Project each vertex to image space to sample depth
        mins = verts.min(axis=0)
        maxs = verts.max(axis=0)
        extents = maxs - mins
        extents[extents < 1e-6] = 1.0

        h, w = depth_map.shape

        # Normalize vertices to [0, 1] range → map to depth map pixel coords
        # Assume mesh is roughly aligned: X=width, Y=height, Z=depth
        uv_x = ((verts[:, 0] - mins[0]) / extents[0] * (w - 1)).astype(int).clip(0, w - 1)
        uv_y = ((verts[:, 1] - mins[1]) / extents[1] * (h - 1)).astype(int).clip(0, h - 1)
        # Flip Y because image coordinates are top-down
        uv_y = (h - 1) - uv_y

        # Sample depth for each vertex
        sampled_depth = depth_map[uv_y, uv_x]

        # Compute displacement: vertices closer to camera get pushed out more
        # Invert depth (0=near → 1 displacement, 1=far → 0 displacement)
        displacement = (1.0 - sampled_depth) * weight * depth_ratio

        # Scale displacement by mesh extent (Z axis = depth)
        z_extent = extents[2] if extents[2] > 1e-6 else extents.max()
        displacement_3d = normals * displacement[:, np.newaxis] * z_extent

        # Apply displacement
        enhanced_verts = verts + displacement_3d

        # Create enhanced mesh
        enhanced_mesh = trimesh.Trimesh(
            vertices=enhanced_verts,
            faces=mesh.faces.copy(),
            process=True,
        )

        # Copy vertex colors if present
        if hasattr(mesh, 'visual') and mesh.visual is not None:
            try:
                enhanced_mesh.visual = mesh.visual.copy()
            except Exception:
                pass

        # Sanity check: enhanced mesh should be roughly same size
        original_vol = mesh.volume if mesh.is_watertight else 0
        enhanced_vol = enhanced_mesh.volume if enhanced_mesh.is_watertight else 0
        if original_vol > 0 and enhanced_vol > 0:
            vol_change = abs(enhanced_vol - original_vol) / original_vol
            if vol_change > 2.0:
                warnings.append(
                    f"Depth enhancement caused {vol_change*100:.0f}% volume change — "
                    f"reverting to original mesh"
                )
                return mesh, warnings

        vert_delta = np.linalg.norm(enhanced_verts - verts, axis=1)
        logger.info(
            f"Depth enhancement applied: mean displacement={vert_delta.mean():.4f}, "
            f"max={vert_delta.max():.4f}, verts={len(verts)}"
        )

        return enhanced_mesh, warnings

    def _light_repair(self, mesh):
        """Light mesh repair that preserves original geometry.

        For high-quality API outputs (Tripo, TRELLIS), voxel-based watertight
        conversion destroys detail. Instead, do minimal fixes:
        - Fix normals
        - Fill small holes
        - Remove degenerate faces
        Does NOT rebuid geometry via marching cubes.
        """
        import trimesh

        logger.info(f"Light repair: {len(mesh.vertices)} verts, watertight={mesh.is_watertight}")

        # Save original visual before any repair
        original_visual = mesh.visual.copy() if hasattr(mesh.visual, 'copy') else mesh.visual

        # Fix normals
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_inversion(mesh)

        # Fill holes
        trimesh.repair.fill_holes(mesh)

        # Restore visual if repair destroyed it
        if (hasattr(original_visual, 'kind') and original_visual.kind == 'texture'
                and (not hasattr(mesh.visual, 'kind') or mesh.visual.kind != 'texture')):
            try:
                mesh.visual = original_visual
                logger.info("Restored texture visual after repair")
            except Exception:
                pass

        logger.info(
            f"Light repair done: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, "
            f"watertight={mesh.is_watertight}"
        )
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
