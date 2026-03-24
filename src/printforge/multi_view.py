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
        """Zero123++ novel view synthesis. Requires zero123plus package."""
        raise NotImplementedError(
            "Zero123++ backend not yet implemented. "
            "Install zero123plus and implement diffusion-based view synthesis."
        )

    def _sv3d_views(self, image_path: str) -> List[Image.Image]:
        """SV3D novel view synthesis. Requires sv3d package."""
        raise NotImplementedError(
            "SV3D backend not yet implemented. "
            "Install sv3d and implement video-diffusion-based view synthesis."
        )
