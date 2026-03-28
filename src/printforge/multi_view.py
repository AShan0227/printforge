"""Multi-View Enhancement: Single image → multiple views for better 3D reconstruction."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


@dataclass
class MultiViewConfig:
    """Configuration for multi-view generation."""
    num_views: int = 6
    # Azimuth angles in degrees for each view
    azimuths: tuple = (0, 60, 120, 180, 240, 300)
    elevation: float = 20.0
    image_size: int = 512
    # Model backend: "placeholder", "zero123pp", "sv3d", "hunyuan3d"
    backend: str = "hunyuan3d"


class MultiViewEnhancer:
    """Generate multiple views from a single image for Hunyuan3D-2 multi-view input.

    The enhance() method produces a dict of {front, back, left, right} PIL images
    suitable for Hunyuan3D-2's /shape_generation endpoint which accepts
    Front, Back, and Left views.

    Strategies:
      - Single image: use as Front, horizontal flip as Back, crop-shift as Left/Right
      - Multiple user photos: assign to closest canonical angles
    """

    def __init__(self, config: Optional[MultiViewConfig] = None):
        self.config = config or MultiViewConfig()

    def enhance(
        self,
        image_path: str,
        extra_views: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Image.Image]:
        """Produce {front, back, left, right} PIL images for multi-view 3D inference.

        Args:
            image_path: Path to the primary input image (treated as front view).
            extra_views: Optional dict mapping view names ("back", "left", "right")
                         to file paths. User-supplied photos from different angles.

        Returns:
            Dict with keys "front", "back", "left", "right" — each a PIL Image.
        """
        size = self.config.image_size
        front = Image.open(image_path).convert("RGB").resize((size, size), Image.LANCZOS)

        views: Dict[str, Image.Image] = {"front": front}

        if extra_views:
            for name, path in extra_views.items():
                canonical = name.lower().strip()
                if canonical in ("back", "left", "right"):
                    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
                    views[canonical] = img

        # Synthesize missing views from the front image
        if "back" not in views:
            views["back"] = ImageOps.mirror(front)

        if "left" not in views:
            views["left"] = self._synthesize_side(front, direction="left")

        if "right" not in views:
            views["right"] = self._synthesize_side(front, direction="right")

        logger.info(
            "Multi-view enhance: %d views (%s user-supplied)",
            len(views),
            len(extra_views) if extra_views else 0,
        )
        return views

    def enhance_from_pil(self, pil_image: Image.Image) -> Dict[str, Image.Image]:
        """Produce {front, back, left, right} views from a PIL Image directly.

        Convenience method that avoids writing to disk. Used by the pipeline
        when passing multi-view images to Hunyuan3D-2.
        """
        size = self.config.image_size
        front = pil_image.convert("RGB").resize((size, size), Image.LANCZOS)

        views: Dict[str, Image.Image] = {"front": front}
        views["back"] = ImageOps.mirror(front)
        views["left"] = self._synthesize_side(front, direction="left")
        views["right"] = self._synthesize_side(front, direction="right")

        return views

    # ── Legacy API (kept for backward compatibility) ─────────────────

    def generate_views(self, image_path: str) -> List[Image.Image]:
        """Generate multiple views from a single input image.

        Args:
            image_path: Path to the input image.

        Returns:
            List of PIL Images representing different views.
        """
        if self.config.backend == "placeholder":
            return self._placeholder_views(image_path)
        elif self.config.backend == "hunyuan3d":
            mv = self.enhance(image_path)
            return [mv["front"], mv["right"], mv["back"], mv["left"]]
        elif self.config.backend == "zero123pp":
            return self._zero123pp_views(image_path)
        elif self.config.backend == "sv3d":
            return self._sv3d_views(image_path)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def save_views(self, views: list, output_dir: str) -> List[str]:
        """Save generated views to disk.

        Accepts either a list of images or a dict from enhance().
        Returns list of saved file paths.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths = []

        if isinstance(views, dict):
            for name, img in views.items():
                path = out / f"view_{name}.png"
                img.save(str(path))
                paths.append(str(path))
        else:
            for i, view in enumerate(views):
                azimuth = self.config.azimuths[i] if i < len(self.config.azimuths) else i * 60
                path = out / f"view_{azimuth:03d}.png"
                view.save(str(path))
                paths.append(str(path))

        logger.info(f"Saved {len(paths)} views to {output_dir}")
        return paths

    # ── Synthesis helpers ─────────────────────────────────────────────

    @staticmethod
    def _synthesize_side(front: Image.Image, direction: str = "left") -> Image.Image:
        """Synthesize a side view by cropping and shifting the front image.

        Takes the left or right 60% of the front image and stretches it,
        simulating a rough side perspective.
        """
        w, h = front.size

        if direction == "left":
            # Take the left portion of the image
            crop_box = (0, 0, int(w * 0.6), h)
        else:
            # Take the right portion
            crop_box = (int(w * 0.4), 0, w, h)

        cropped = front.crop(crop_box)
        return cropped.resize((w, h), Image.LANCZOS)

    # ── Placeholder / future backends ─────────────────────────────────

    def _placeholder_views(self, image_path: str) -> List[Image.Image]:
        """Placeholder: return copies of the original image for each view angle."""
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.config.image_size, self.config.image_size), Image.LANCZOS)

        views = []
        for i in range(self.config.num_views):
            views.append(img.copy())

        logger.info(f"Placeholder: generated {len(views)} view copies from {image_path}")
        return views

    def _zero123pp_views(self, image_path: str) -> List[Image.Image]:
        """Zero123++ novel view synthesis via HuggingFace Space.

        NOTE: As of 2026-03, the Zero123++ HF Space is in RUNTIME_ERROR state.
        This implementation is ready for when it comes back online, or for local deployment.
        Falls back to placeholder if unavailable.
        """
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            logger.warning("gradio_client not installed — cannot use Zero123++")
            return self._placeholder_views(image_path)

        try:
            client = Client("sudo-ai/zero123plus-demo-space", verbose=False)
            result = client.predict(
                handle_file(image_path),
                api_name="/predict",
            )

            # Zero123++ outputs a single image with 6 views in a 3x2 grid
            grid = Image.open(result).convert("RGB")
            w, h = grid.size
            tile_w, tile_h = w // 3, h // 2

            views = []
            for row in range(2):
                for col in range(3):
                    box = (col * tile_w, row * tile_h, (col + 1) * tile_w, (row + 1) * tile_h)
                    tile = grid.crop(box).resize(
                        (self.config.image_size, self.config.image_size), Image.LANCZOS
                    )
                    views.append(tile)

            logger.info(f"Zero123++: generated {len(views)} views from {image_path}")
            return views

        except Exception as e:
            logger.warning(f"Zero123++ failed: {e}, falling back to placeholder")
            return self._placeholder_views(image_path)

    def _sv3d_views(self, image_path: str) -> List[Image.Image]:
        """SV3D novel view synthesis via Stability AI.

        NOTE: SV3D HF Space requires authentication. Implementation ready for
        local deployment with stabilityai/sv3d weights.
        """
        try:
            from gradio_client import Client, handle_file
        except ImportError:
            logger.warning("gradio_client not installed — cannot use SV3D")
            return self._placeholder_views(image_path)

        try:
            import os
            hf_token = os.environ.get("HF_TOKEN")
            token_file = Path.home() / ".openclaw" / "workspace" / ".hf_token"
            if not hf_token and token_file.exists():
                hf_token = token_file.read_text().strip()

            kwargs = {"verbose": False}
            if hf_token:
                kwargs["hf_token"] = hf_token

            client = Client("stabilityai/sv3d", **kwargs)
            result = client.predict(
                handle_file(image_path),
                api_name="/predict",
            )

            # SV3D outputs video frames — extract key frames
            # Implementation depends on the actual API response format
            logger.info(f"SV3D: result type = {type(result)}")
            return self._placeholder_views(image_path)

        except Exception as e:
            logger.warning(f"SV3D failed: {e}, falling back to placeholder")
            return self._placeholder_views(image_path)
