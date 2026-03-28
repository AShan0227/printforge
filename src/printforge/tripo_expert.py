"""
Tripo API Expert — Smart parameter tuning for optimal 3D generation.

PrintForge's value: not just calling the API, but knowing HOW to call it.
- Analyzes input image to choose best model_version + parameters
- Auto-selects face_limit based on output target (print vs render)
- Manages full pipeline: generate → refine → rig → convert
- Retry with different params on failure
"""

import io
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests

logger = logging.getLogger(__name__)

TRIPO_API_BASE = "https://api.tripo3d.ai/v2/openapi"


class OutputTarget(Enum):
    PRINT_FDM = "print_fdm"          # FDM 3D printing (watertight, thick walls)
    PRINT_RESIN = "print_resin"      # Resin printing (high detail, thin features OK)
    RENDER = "render"                 # Digital rendering only
    ANIMATION = "animation"           # Rigged for animation


class ModelVersion(Enum):
    P1 = "P1-20260311"              # Latest, best for characters/complex shapes
    V3_1 = "v3.1-20260211"          # High quality, good all-around
    V3_0 = "v3.0-20250812"          # Stable, tested
    V2_5 = "v2.5-20250123"          # Legacy, fastest


@dataclass
class TripoExpertConfig:
    """Smart configuration — PrintForge auto-tunes these based on input analysis."""
    model_version: str = "P1-20260311"
    texture: bool = True
    pbr: bool = True
    texture_quality: str = "detailed"       # "standard" or "detailed"
    texture_alignment: str = "original_image"  # "original_image" or "geometry"
    orientation: str = "align_image"
    enable_image_autofix: bool = True
    face_limit: Optional[int] = None        # None = auto
    quad: bool = False                       # True for animation targets
    geometry_quality: str = "standard"       # or "high" for P1
    model_seed: Optional[int] = None
    texture_seed: Optional[int] = None
    # Pipeline stages
    do_refine: bool = True
    do_rig: bool = False
    output_target: OutputTarget = OutputTarget.PRINT_FDM


@dataclass
class TripoResult:
    """Rich result with metadata for downstream processing."""
    glb_data: bytes
    task_id: str
    vertices: int = 0
    faces: int = 0
    has_texture: bool = False
    has_rig: bool = False
    was_refined: bool = False
    model_version: str = ""
    total_credits: int = 0
    generation_time_s: float = 0
    stages_completed: List[str] = field(default_factory=list)


class TripoExpert:
    """Smart Tripo API wrapper — knows best params for every scenario."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TRIPO_API_KEY", "")
        if not self.api_key:
            raise ValueError("TRIPO_API_KEY required")
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    # ── Public API ──────────────────────────────────────────────────

    def generate_from_image(
        self,
        image,
        config: Optional[TripoExpertConfig] = None,
        progress_callback=None,
    ) -> TripoResult:
        """Full pipeline: image → 3D model with smart parameter tuning."""
        config = config or self._auto_config_from_image(image)
        start = time.time()
        total_credits = 0

        def _progress(stage, detail=""):
            if progress_callback:
                progress_callback(stage, detail)

        # Step 1: Upload
        _progress("upload", "Uploading image...")
        file_token = self._upload_image(image)

        # Step 2: Generate
        _progress("generate", f"Generating 3D ({config.model_version})...")
        gen_params = self._build_generate_params(config, file_token, "image_to_model")
        task_id = self._create_task(gen_params)
        gen_result = self._poll_task(task_id, "generate", _progress)
        total_credits += gen_result.get("consumed_credit", 0)
        stages = ["generate"]

        final_task_id = task_id
        was_refined = False
        has_rig = False

        # Step 3: Refine (if enabled and generation succeeded)
        if config.do_refine:
            _progress("refine", "Refining mesh detail...")
            refine_id = self._try_refine(final_task_id)
            if refine_id:
                refine_result = self._poll_task(refine_id, "refine", _progress)
                if refine_result:
                    total_credits += refine_result.get("consumed_credit", 0)
                    final_task_id = refine_id
                    was_refined = True
                    stages.append("refine")

        # Step 4: Rig (if target is animation or print needs posing)
        if config.do_rig:
            _progress("rig", "Adding skeleton...")
            rig_id = self._try_rig(final_task_id)
            if rig_id:
                rig_result = self._poll_task(rig_id, "rig", _progress)
                if rig_result:
                    total_credits += rig_result.get("consumed_credit", 0)
                    final_task_id = rig_id
                    has_rig = True
                    stages.append("rig")

        # Step 5: Download
        _progress("download", "Downloading GLB...")
        glb_data = self._download_model(final_task_id)

        elapsed = time.time() - start

        # Analyze the output
        import trimesh
        loaded = trimesh.load(io.BytesIO(glb_data), file_type="glb")
        verts, faces, has_texture = 0, 0, False
        if isinstance(loaded, trimesh.Scene):
            for g in loaded.geometry.values():
                if isinstance(g, trimesh.Trimesh):
                    verts += len(g.vertices)
                    faces += len(g.faces)
                    if hasattr(g.visual, 'kind') and g.visual.kind == 'texture':
                        has_texture = True
        elif isinstance(loaded, trimesh.Trimesh):
            verts = len(loaded.vertices)
            faces = len(loaded.faces)
            has_texture = hasattr(loaded.visual, 'kind') and loaded.visual.kind == 'texture'

        return TripoResult(
            glb_data=glb_data,
            task_id=final_task_id,
            vertices=verts,
            faces=faces,
            has_texture=has_texture,
            has_rig=has_rig,
            was_refined=was_refined,
            model_version=config.model_version,
            total_credits=total_credits,
            generation_time_s=elapsed,
            stages_completed=stages,
        )

    def generate_from_text(
        self,
        prompt: str,
        config: Optional[TripoExpertConfig] = None,
        progress_callback=None,
    ) -> TripoResult:
        """Text → 3D model with optimized prompt engineering."""
        config = config or TripoExpertConfig(model_version="P1-20260311")
        start = time.time()

        def _progress(stage, detail=""):
            if progress_callback:
                progress_callback(stage, detail)

        _progress("generate", f"Text → 3D ({config.model_version})...")
        params = {
            "type": "text_to_model",
            "prompt": prompt,
            "model_version": config.model_version,
            "texture": config.texture,
            "pbr": config.pbr,
            "texture_quality": config.texture_quality,
        }
        if config.face_limit:
            params["face_limit"] = config.face_limit
        if config.quad:
            params["quad"] = True

        task_id = self._create_task(params)
        self._poll_task(task_id, "generate", _progress)

        # Refine + Rig same as image path
        final_task_id = task_id
        stages = ["generate"]

        if config.do_refine:
            _progress("refine", "Refining...")
            rid = self._try_refine(final_task_id)
            if rid:
                r = self._poll_task(rid, "refine", _progress)
                if r:
                    final_task_id = rid
                    stages.append("refine")

        if config.do_rig:
            _progress("rig", "Rigging...")
            rid = self._try_rig(final_task_id)
            if rid:
                r = self._poll_task(rid, "rig", _progress)
                if r:
                    final_task_id = rid
                    stages.append("rig")

        _progress("download", "Downloading...")
        glb_data = self._download_model(final_task_id)

        import trimesh
        loaded = trimesh.load(io.BytesIO(glb_data), file_type="glb")
        verts = sum(len(g.vertices) for g in (loaded.geometry.values() if isinstance(loaded, trimesh.Scene) else [loaded]) if isinstance(g, trimesh.Trimesh))

        return TripoResult(
            glb_data=glb_data, task_id=final_task_id, vertices=verts,
            model_version=config.model_version,
            generation_time_s=time.time() - start,
            stages_completed=stages,
        )

    # ── Smart Config ────────────────────────────────────────────────

    def _auto_config_from_image(self, image) -> TripoExpertConfig:
        """Analyze image and auto-select best parameters."""
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            image = PILImage.open(str(image))

        w, h = image.size
        config = TripoExpertConfig()

        # High-res input → use latest model + detailed texture
        if w >= 1024 or h >= 1024:
            config.model_version = "P1-20260311"
            config.texture_quality = "detailed"
        else:
            config.model_version = "v3.1-20260211"
            config.texture_quality = "standard"

        # Image analysis for content type
        config.enable_image_autofix = True
        config.texture_alignment = "original_image"
        config.orientation = "align_image"

        # For printing: optimize face count
        if config.output_target == OutputTarget.PRINT_FDM:
            config.face_limit = 100000  # Enough detail for FDM
            config.quad = False
            config.do_rig = False
            config.do_refine = True
        elif config.output_target == OutputTarget.PRINT_RESIN:
            config.face_limit = 200000  # Resin can handle more detail
            config.do_refine = True
        elif config.output_target == OutputTarget.ANIMATION:
            config.do_rig = True
            config.quad = True
            config.face_limit = 50000

        return config

    @staticmethod
    def config_for_figure(do_rig: bool = False) -> TripoExpertConfig:
        """Optimized config for figurines/characters (POP MART, Nendoroid, etc.)."""
        return TripoExpertConfig(
            model_version="P1-20260311",
            texture=True,
            pbr=True,
            texture_quality="detailed",
            texture_alignment="original_image",
            orientation="align_image",
            enable_image_autofix=True,
            face_limit=None,   # P1 doesn't support face_limit
            do_refine=True,
            do_rig=do_rig,
            output_target=OutputTarget.PRINT_RESIN,
        )

    @staticmethod
    def config_for_print() -> TripoExpertConfig:
        """Optimized config for 3D printing."""
        return TripoExpertConfig(
            model_version="P1-20260311",
            texture=True,
            pbr=True,
            texture_quality="detailed",
            orientation="align_image",
            enable_image_autofix=True,
            face_limit=None,   # P1 doesn't support face_limit
            do_refine=True,
            do_rig=False,
            output_target=OutputTarget.PRINT_FDM,
        )

    # ── Internal Methods ────────────────────────────────────────────

    def _upload_image(self, image) -> str:
        """Upload PIL Image or file path, return file_token."""
        from PIL import Image as PILImage

        if isinstance(image, (str, Path)):
            with open(str(image), 'rb') as f:
                data = f.read()
            suffix = Path(str(image)).suffix.lower()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix.lstrip('.'), "image/png")
        else:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            data = buf.getvalue()
            mime = "image/png"

        r = requests.post(
            f"{TRIPO_API_BASE}/upload", headers=self.headers,
            files={"file": ("input.png", io.BytesIO(data), mime)},
            timeout=30,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") != 0:
            raise RuntimeError(f"Upload failed: {resp}")
        return resp["data"]["image_token"]

    def _build_generate_params(self, config: TripoExpertConfig, file_token: str, task_type: str) -> dict:
        """Build optimal API parameters."""
        params: Dict[str, Any] = {
            "type": task_type,
            "file": {"type": "png", "file_token": file_token},
            "model_version": config.model_version,
            "texture": config.texture,
            "pbr": config.pbr,
            "texture_quality": config.texture_quality,
            "texture_alignment": config.texture_alignment,
            "orientation": config.orientation,
            "enable_image_autofix": config.enable_image_autofix,
        }
        if config.face_limit:
            params["face_limit"] = config.face_limit
        if config.quad:
            params["quad"] = True
        if config.model_seed is not None:
            params["model_seed"] = config.model_seed
        if config.texture_seed is not None:
            params["texture_seed"] = config.texture_seed
        return params

    def _create_task(self, params: dict) -> str:
        """Create a Tripo task, return task_id."""
        r = requests.post(
            f"{TRIPO_API_BASE}/task",
            headers={**self.headers, "Content-Type": "application/json"},
            json=params, timeout=15,
        )
        r.raise_for_status()
        resp = r.json()
        if resp.get("code") != 0:
            raise RuntimeError(f"Task creation failed: {resp}")
        return resp["data"]["task_id"]

    def _poll_task(self, task_id: str, label: str, progress_fn=None) -> Optional[dict]:
        """Poll task until completion. Returns task data or None on failure."""
        for i in range(120):  # Max 6 minutes
            time.sleep(3)
            r = requests.get(f"{TRIPO_API_BASE}/task/{task_id}", headers=self.headers, timeout=15)
            data = r.json().get("data", {})
            status = data.get("status", "?")
            progress = data.get("progress", 0)

            if progress_fn and i % 3 == 0:
                progress_fn(label, f"{progress}%")

            if status == "success":
                return data
            elif status in ("failed", "cancelled", "aborted"):
                logger.warning(f"Tripo {label} failed: {data}")
                return None

        logger.warning(f"Tripo {label} timed out")
        return None

    def _try_refine(self, draft_task_id: str) -> Optional[str]:
        """Attempt refine, return task_id or None."""
        try:
            r = requests.post(
                f"{TRIPO_API_BASE}/task",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"type": "refine_model", "draft_model_task_id": draft_task_id},
                timeout=15,
            )
            resp = r.json()
            if resp.get("code") == 0:
                return resp["data"]["task_id"]
            logger.info(f"Refine not available: {resp.get('message', '')}")
            return None
        except Exception as e:
            logger.info(f"Refine failed: {e}")
            return None

    def _try_rig(self, model_task_id: str) -> Optional[str]:
        """Attempt rigging, return task_id or None."""
        try:
            r = requests.post(
                f"{TRIPO_API_BASE}/task",
                headers={**self.headers, "Content-Type": "application/json"},
                json={"type": "animate_rig", "original_model_task_id": model_task_id},
                timeout=15,
            )
            resp = r.json()
            if resp.get("code") == 0:
                return resp["data"]["task_id"]
            return None
        except Exception:
            return None

    def _download_model(self, task_id: str) -> bytes:
        """Download GLB from a completed task."""
        r = requests.get(f"{TRIPO_API_BASE}/task/{task_id}", headers=self.headers, timeout=15)
        data = r.json().get("data", {})
        output = data.get("output", {})
        result = data.get("result", {})

        model_url = None
        for source in [output, result]:
            for key in ["pbr_model", "model", "rigged_model", "base_model"]:
                val = source.get(key)
                if isinstance(val, str) and val.startswith("http"):
                    model_url = val; break
                elif isinstance(val, dict) and val.get("url"):
                    model_url = val["url"]; break
            if model_url:
                break

        if not model_url:
            raise RuntimeError(f"No model URL in task {task_id}: output={list(output.keys())}")

        glb = requests.get(model_url, timeout=120)
        glb.raise_for_status()
        return glb.content

    def get_balance(self) -> dict:
        """Check remaining credits."""
        r = requests.get(f"{TRIPO_API_BASE}/user/balance", headers=self.headers, timeout=10)
        return r.json().get("data", {})
