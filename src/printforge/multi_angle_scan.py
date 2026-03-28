"""
Multi-Angle Scan — Upload multiple photos from different angles for high-quality reconstruction.

Workflow:
  1. User takes 3-8 photos around the object (phone camera)
  2. Upload all photos
  3. Auto-classify view angles (front/back/left/right/top)
  4. Submit to Tripo multiview API or compose reference sheet
  5. Generate high-quality 3D with maximum context
"""

import io
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ViewImage:
    """A single view image with metadata."""
    path: str
    view_angle: str = "unknown"  # front, back, left, right, top, 3/4, unknown
    confidence: float = 0
    width: int = 0
    height: int = 0


@dataclass
class ScanSession:
    """Collection of multi-angle images for one object."""
    views: List[ViewImage] = field(default_factory=list)
    has_front: bool = False
    has_back: bool = False
    has_left: bool = False
    has_right: bool = False
    coverage_score: float = 0  # 0-1, how well covered the object is


class MultiAngleScanner:
    """Process multi-angle photos for high-quality 3D reconstruction."""

    # Minimum views for good quality
    MIN_VIEWS = 2
    IDEAL_VIEWS = 4  # front + back + left + right

    def create_session(self, image_paths: List[str]) -> ScanSession:
        """Create a scan session from multiple images.

        Auto-classifies view angles based on image order and content.
        """
        from PIL import Image

        session = ScanSession()

        for i, path in enumerate(image_paths):
            img = Image.open(path)
            w, h = img.size

            # Heuristic angle classification based on order
            # Most people photograph: front → right → back → left
            if len(image_paths) >= 4:
                angle_map = {0: "front", 1: "right", 2: "back", 3: "left"}
                angle = angle_map.get(i, "unknown")
            elif len(image_paths) == 3:
                angle_map = {0: "front", 1: "side", 2: "back"}
                angle = angle_map.get(i, "unknown")
            elif len(image_paths) == 2:
                angle = "front" if i == 0 else "back"
            else:
                angle = "front"

            view = ViewImage(
                path=path,
                view_angle=angle,
                confidence=0.7 if len(image_paths) >= 4 else 0.5,
                width=w,
                height=h,
            )
            session.views.append(view)

        # Update coverage
        angles = {v.view_angle for v in session.views}
        session.has_front = "front" in angles
        session.has_back = "back" in angles
        session.has_left = "left" in angles or "side" in angles
        session.has_right = "right" in angles or "side" in angles
        session.coverage_score = len(angles & {"front", "back", "left", "right"}) / 4.0

        logger.info(
            f"Scan session: {len(session.views)} views, "
            f"coverage={session.coverage_score:.0%}, "
            f"angles={[v.view_angle for v in session.views]}"
        )
        return session

    def generate_from_scan(
        self,
        session: ScanSession,
        output_path: str,
        progress_callback=None,
    ) -> dict:
        """Generate 3D model from multi-angle scan session.

        Uses Tripo multiview_to_model when possible.
        """
        import requests
        import trimesh

        tripo_key = os.environ.get("TRIPO_API_KEY", "")
        if not tripo_key:
            raise RuntimeError("TRIPO_API_KEY required for multi-angle generation")

        headers = {"Authorization": f"Bearer {tripo_key}"}
        base = "https://api.tripo3d.ai/v2/openapi"

        def _progress(stage, detail=""):
            if progress_callback:
                progress_callback(stage, detail)

        # Upload all images
        _progress("upload", f"Uploading {len(session.views)} views...")
        tokens = {}
        for view in session.views:
            with open(view.path, 'rb') as f:
                r = requests.post(
                    f"{base}/upload", headers=headers,
                    files={"file": (Path(view.path).name, f, "image/jpeg")},
                    timeout=15,
                )
                r.raise_for_status()
                tokens[view.view_angle] = r.json()["data"]["image_token"]

        # Build multiview task
        files = []
        # Front must be first
        if "front" in tokens:
            files.append({"type": "jpg", "file_token": tokens["front"]})
        else:
            # Use first image as front
            first_angle = session.views[0].view_angle
            files.append({"type": "jpg", "file_token": tokens[first_angle]})

        # Add other views
        for angle in ["left", "back", "right"]:
            if angle in tokens and angle != session.views[0].view_angle:
                files.append({"type": "jpg", "file_token": tokens[angle]})

        # Also add any remaining views
        for view in session.views:
            if view.view_angle not in ["front", "left", "back", "right"]:
                if view.view_angle in tokens:
                    files.append({"type": "jpg", "file_token": tokens[view.view_angle]})

        _progress("generate", f"Multiview generation ({len(files)} views)...")

        import time
        r2 = requests.post(
            f"{base}/task",
            headers={**headers, "Content-Type": "application/json"},
            json={"type": "multiview_to_model", "files": files},
            timeout=15,
        )
        r2.raise_for_status()
        resp = r2.json()
        if resp.get("code") != 0:
            raise RuntimeError(f"Multiview task failed: {resp}")

        task_id = resp["data"]["task_id"]

        # Poll
        for i in range(90):
            time.sleep(2)
            poll = requests.get(f"{base}/task/{task_id}", headers=headers, timeout=10)
            data = poll.json().get("data", {})
            status = data.get("status", "?")
            if i % 5 == 0:
                _progress("generating", f"{data.get('progress', 0)}%")
            if status == "success":
                break
            elif status in ("failed", "cancelled"):
                raise RuntimeError(f"Multiview generation failed: {status}")

        # Download
        _progress("download", "Downloading GLB...")
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
            raise RuntimeError("No model URL in response")

        glb = requests.get(model_url, timeout=60)
        with open(output_path, 'wb') as f:
            f.write(glb.content)

        # Analyze
        loaded = trimesh.load(io.BytesIO(glb.content), file_type="glb")
        verts = sum(
            len(g.vertices) for g in
            (loaded.geometry.values() if isinstance(loaded, trimesh.Scene) else [loaded])
            if isinstance(g, trimesh.Trimesh)
        )

        return {
            "output_path": output_path,
            "vertices": verts,
            "file_size": len(glb.content),
            "views_used": len(files),
            "coverage": session.coverage_score,
            "credits": data.get("consumed_credit", 0),
        }

    def create_reference_sheet(
        self,
        session: ScanSession,
        output_path: str,
        cols: int = 4,
    ) -> str:
        """Compose all views into a single reference sheet image."""
        from PIL import Image

        images = []
        for view in session.views:
            img = Image.open(view.path).convert("RGB")
            # Normalize to same height
            target_h = 512
            ratio = target_h / img.height
            img = img.resize((int(img.width * ratio), target_h), Image.LANCZOS)
            images.append((img, view.view_angle))

        if not images:
            raise ValueError("No images in session")

        # Calculate grid
        n = len(images)
        rows = (n + cols - 1) // cols
        max_w = max(img.width for img, _ in images)
        label_h = 30

        sheet = Image.new("RGB", (max_w * cols, (512 + label_h) * rows), (30, 30, 30))

        from PIL import ImageDraw
        draw = ImageDraw.Draw(sheet)

        for i, (img, angle) in enumerate(images):
            row, col = divmod(i, cols)
            x = col * max_w + (max_w - img.width) // 2
            y = row * (512 + label_h)
            sheet.paste(img, (x, y))
            # Label
            draw.text((col * max_w + 5, y + 512 + 5), angle.upper(), fill=(200, 200, 200))

        sheet.save(output_path)
        logger.info(f"Reference sheet saved: {output_path} ({sheet.size})")
        return output_path
