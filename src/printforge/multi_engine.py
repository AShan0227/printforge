"""
Multi-Engine Comparison — Run multiple 3D generation backends in parallel,
let the user pick the best result.

Supported engines:
  - Tripo API (P1, v3.1, v2.5)
  - TRELLIS (HF Space)
  - Future: Meshy, Rodin, etc.

Core value: users can't easily compare engines. We do it for them.
"""

import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class EngineResult:
    """Result from a single engine."""
    engine: str
    success: bool
    glb_data: Optional[bytes] = None
    vertices: int = 0
    faces: int = 0
    has_texture: bool = False
    generation_time_s: float = 0
    credits_used: int = 0
    error: Optional[str] = None
    quality_score: float = 0  # 0-1, computed after generation

    @property
    def file_size(self) -> int:
        return len(self.glb_data) if self.glb_data else 0


@dataclass
class ComparisonResult:
    """Multi-engine comparison output."""
    results: Dict[str, EngineResult] = field(default_factory=dict)
    recommended: Optional[str] = None
    total_time_s: float = 0
    total_credits: int = 0

    @property
    def successful(self) -> List[EngineResult]:
        return [r for r in self.results.values() if r.success]


class MultiEngine:
    """Compare multiple 3D generation engines on the same input."""

    ENGINES = {
        "tripo_p1": {
            "name": "Tripo P1 (Latest)",
            "model_version": "P1-20260311",
            "credits": 50,
        },
        "tripo_v3": {
            "name": "Tripo v3.1",
            "model_version": "v3.1-20260211",
            "credits": 30,
        },
        "tripo_v2": {
            "name": "Tripo v2.5 (Fast)",
            "model_version": "v2.5-20250123",
            "credits": 20,
        },
    }

    def __init__(self):
        self.tripo_key = os.environ.get("TRIPO_API_KEY", "")

    def compare(
        self,
        image,
        engines: Optional[List[str]] = None,
        progress_callback: Optional[Callable] = None,
        max_parallel: int = 2,
    ) -> ComparisonResult:
        """Run multiple engines on the same image and compare results.

        Args:
            image: PIL Image or file path
            engines: List of engine IDs to use (default: all available)
            progress_callback: Optional(engine, stage, detail)
            max_parallel: Max concurrent API calls

        Returns:
            ComparisonResult with all engine outputs + recommendation
        """
        from PIL import Image as PILImage

        if isinstance(image, str):
            image = PILImage.open(image)

        if engines is None:
            engines = list(self.ENGINES.keys())

        # Filter to available engines
        available = []
        for eng in engines:
            if eng.startswith("tripo") and self.tripo_key:
                available.append(eng)
            # Future: add TRELLIS, Meshy checks

        if not available:
            logger.warning("No engines available")
            return ComparisonResult()

        start = time.time()
        results = {}

        def _progress(engine, stage, detail=""):
            if progress_callback:
                progress_callback(engine, stage, detail)

        # Run engines in parallel
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {}
            for eng in available:
                f = executor.submit(self._run_engine, eng, image, _progress)
                futures[f] = eng

            for future in as_completed(futures):
                eng = futures[future]
                try:
                    result = future.result()
                    results[eng] = result
                except Exception as e:
                    results[eng] = EngineResult(
                        engine=eng, success=False, error=str(e)
                    )

        # Score and rank results
        comparison = ComparisonResult(
            results=results,
            total_time_s=time.time() - start,
            total_credits=sum(r.credits_used for r in results.values()),
        )

        self._score_results(comparison)
        return comparison

    def _run_engine(
        self, engine_id: str, image, progress_fn
    ) -> EngineResult:
        """Run a single engine."""
        config = self.ENGINES.get(engine_id)
        if not config:
            return EngineResult(engine=engine_id, success=False, error="Unknown engine")

        start = time.time()
        progress_fn(engine_id, "starting", config["name"])

        if engine_id.startswith("tripo"):
            return self._run_tripo(engine_id, config, image, progress_fn)
        else:
            return EngineResult(engine=engine_id, success=False, error="Not implemented")

    def _run_tripo(self, engine_id, config, image, progress_fn) -> EngineResult:
        """Run Tripo with specific model version."""
        import requests
        import trimesh

        headers = {"Authorization": f"Bearer {self.tripo_key}"}
        base = "https://api.tripo3d.ai/v2/openapi"
        start = time.time()

        try:
            # Upload
            progress_fn(engine_id, "upload", "")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            r = requests.post(f"{base}/upload", headers=headers,
                              files={"file": ("img.png", buf, "image/png")}, timeout=15)
            r.raise_for_status()
            token = r.json()["data"]["image_token"]

            # Generate
            progress_fn(engine_id, "generate", config["model_version"])
            params = {
                "type": "image_to_model",
                "file": {"type": "png", "file_token": token},
                "model_version": config["model_version"],
                "texture": True,
                "pbr": True,
                "texture_quality": "detailed",
                "enable_image_autofix": True,
            }
            r2 = requests.post(f"{base}/task",
                               headers={**headers, "Content-Type": "application/json"},
                               json=params, timeout=15)
            r2.raise_for_status()
            task_data = r2.json()
            if task_data.get("code") != 0:
                return EngineResult(engine=engine_id, success=False,
                                    error=task_data.get("message", "Task failed"))
            task_id = task_data["data"]["task_id"]

            # Poll
            for i in range(90):
                time.sleep(2)
                poll = requests.get(f"{base}/task/{task_id}", headers=headers, timeout=10)
                data = poll.json().get("data", {})
                status = data.get("status", "?")
                if i % 5 == 0:
                    progress_fn(engine_id, "generating", f"{data.get('progress', 0)}%")
                if status == "success":
                    break
                elif status in ("failed", "cancelled"):
                    return EngineResult(engine=engine_id, success=False,
                                        error=f"Task {status}",
                                        generation_time_s=time.time() - start)

            # Download
            progress_fn(engine_id, "download", "")
            output = data.get("output", {})
            result_data = data.get("result", {})
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
                return EngineResult(engine=engine_id, success=False, error="No model URL")

            glb = requests.get(model_url, timeout=60)
            glb_data = glb.content

            # Analyze
            loaded = trimesh.load(io.BytesIO(glb_data), file_type="glb")
            verts, faces, has_tex = 0, 0, False
            if isinstance(loaded, trimesh.Scene):
                for g in loaded.geometry.values():
                    if isinstance(g, trimesh.Trimesh):
                        verts += len(g.vertices)
                        faces += len(g.faces)
                        if hasattr(g.visual, 'kind') and g.visual.kind == 'texture':
                            has_tex = True
            else:
                verts = len(loaded.vertices)
                faces = len(loaded.faces)

            credits = data.get("consumed_credit", config.get("credits", 0))

            return EngineResult(
                engine=engine_id,
                success=True,
                glb_data=glb_data,
                vertices=verts,
                faces=faces,
                has_texture=has_tex,
                generation_time_s=time.time() - start,
                credits_used=credits,
            )

        except Exception as e:
            return EngineResult(engine=engine_id, success=False,
                                error=str(e), generation_time_s=time.time() - start)

    def _score_results(self, comparison: ComparisonResult):
        """Score and rank results based on quality heuristics."""
        for result in comparison.results.values():
            if not result.success:
                result.quality_score = 0
                continue

            score = 0.5  # Base

            # More vertices = more detail (up to a point)
            if result.vertices > 10000:
                score += 0.15
            elif result.vertices > 5000:
                score += 0.1
            elif result.vertices > 1000:
                score += 0.05

            # Texture is important
            if result.has_texture:
                score += 0.2

            # File size as proxy for detail
            if result.file_size > 2_000_000:
                score += 0.1
            elif result.file_size > 500_000:
                score += 0.05

            # Speed bonus (faster is better when quality is similar)
            if result.generation_time_s < 60:
                score += 0.05

            result.quality_score = min(score, 1.0)

        # Recommend the highest scoring
        successful = comparison.successful
        if successful:
            best = max(successful, key=lambda r: r.quality_score)
            comparison.recommended = best.engine
