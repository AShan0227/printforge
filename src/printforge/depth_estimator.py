"""
Depth Estimator — Single-image depth estimation for 3D reconstruction.
=====================================================================

Supports:
  - HuggingFace Inference API (DPT-Large / Depth-Anything)
  - Fallback: edge-based gradient depth estimation (CPU, no API needed)
  - Depth map → point cloud → mesh conversion
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from env or file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".openclaw" / "workspace" / ".hf_token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


@dataclass
class DepthResult:
    """Result of depth estimation."""
    depth_map: np.ndarray  # H x W, normalized 0-1 (0=near, 1=far)
    min_depth: float
    max_depth: float
    foreground_mask: np.ndarray  # bool, H x W
    estimated_depth_ratio: float  # depth / width of foreground object


class DepthEstimator:
    """Estimate depth from a single image."""

    # HuggingFace models for depth estimation
    MODELS = [
        "LiheYoung/depth-anything-large-hf",
        "Intel/dpt-large",
    ]

    def __init__(self, model: Optional[str] = None):
        self.model = model or self.MODELS[0]

    def estimate(self, image) -> DepthResult:
        """Estimate depth map from a PIL Image.

        Tries HF API first, falls back to gradient-based estimation.

        Args:
            image: PIL.Image

        Returns:
            DepthResult with depth map and metadata
        """
        from PIL import Image as PILImage

        if not isinstance(image, PILImage.Image):
            image = PILImage.open(str(image)).convert("RGB")
        else:
            image = image.convert("RGB")

        # Try HF API
        depth_map = self._estimate_hf_api(image)

        if depth_map is None:
            logger.info("HF API unavailable, using gradient fallback")
            depth_map = self._estimate_gradient(image)

        # Compute foreground mask (depth < 0.7 of max = foreground)
        threshold = 0.7
        foreground_mask = depth_map < threshold

        # Estimate depth ratio
        if foreground_mask.any():
            fg_depths = depth_map[foreground_mask]
            depth_range = fg_depths.max() - fg_depths.min()
            # Estimate width from mask columns
            cols_with_fg = np.any(foreground_mask, axis=0)
            fg_width = cols_with_fg.sum() / foreground_mask.shape[1]
            depth_ratio = depth_range / max(fg_width, 0.01)
        else:
            depth_ratio = 0.5

        return DepthResult(
            depth_map=depth_map,
            min_depth=float(depth_map.min()),
            max_depth=float(depth_map.max()),
            foreground_mask=foreground_mask,
            estimated_depth_ratio=float(depth_ratio),
        )

    def _estimate_hf_api(self, image) -> Optional[np.ndarray]:
        """Estimate depth via HuggingFace Inference API."""
        import requests
        import io

        hf_token = _get_hf_token()
        if not hf_token:
            logger.info("No HF_TOKEN — skipping HF API depth estimation")
            return None

        try:
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            for model in ([self.model] + self.MODELS):
                url = f"https://api-inference.huggingface.co/models/{model}"
                resp = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {hf_token}"},
                    data=image_bytes,
                    timeout=60,
                )

                if resp.status_code == 200:
                    # Response is a depth image (grayscale PNG)
                    from PIL import Image as PILImage
                    depth_img = PILImage.open(io.BytesIO(resp.content)).convert("L")
                    depth_arr = np.array(depth_img).astype(np.float32) / 255.0
                    logger.info(f"HF API depth estimation OK via {model}")
                    return depth_arr
                else:
                    logger.warning(f"HF API {model}: {resp.status_code}")
                    continue

            return None

        except Exception as e:
            logger.warning(f"HF API depth estimation failed: {e}")
            return None

    @staticmethod
    def _estimate_gradient(image) -> np.ndarray:
        """Fallback: estimate depth from image gradients and edges.

        Uses Sobel-like edge detection + center-distance weighting.
        Objects in center are assumed closer (shallower depth).
        """
        arr = np.array(image.convert("L")).astype(np.float32) / 255.0
        h, w = arr.shape

        # Sobel gradients (manual, no scipy/cv2 dependency)
        gx = np.zeros_like(arr)
        gy = np.zeros_like(arr)
        gx[:, 1:-1] = arr[:, 2:] - arr[:, :-2]
        gy[1:-1, :] = arr[2:, :] - arr[:-2, :]
        edges = np.sqrt(gx ** 2 + gy ** 2)

        # Normalize edges
        edges = edges / max(edges.max(), 1e-6)

        # Center distance map (closer to center = less depth = foreground)
        cy, cx = h / 2, w / 2
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        dist = dist / dist.max()

        # Combine: edges suggest detail (foreground), center suggests nearness
        # Depth = distance_from_center * (1 - edge_strength)
        depth = dist * (1 - 0.5 * edges)

        # Invert intensity: brighter areas in photo tend to be closer
        brightness_depth = 1.0 - arr
        depth = 0.6 * depth + 0.4 * brightness_depth

        # Normalize to [0, 1]
        depth = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)

        return depth


def estimate_depth(image_path: str) -> DepthResult:
    """Convenience function: estimate depth from an image file."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    estimator = DepthEstimator()
    return estimator.estimate(img)


def depth_to_pointcloud(
    depth_map: np.ndarray,
    image,
    fov: float = 60.0,
) -> np.ndarray:
    """Convert depth map + RGB image to colored point cloud.

    Args:
        depth_map: H x W normalized depth (0=near, 1=far)
        image: PIL.Image (RGB)
        fov: Field of view in degrees

    Returns:
        np.ndarray of shape (N, 6): [x, y, z, r, g, b]
    """
    from PIL import Image as PILImage

    img_arr = np.array(image.convert("RGB")).astype(np.float32)
    h, w = depth_map.shape

    # Resize image to match depth map if needed
    if img_arr.shape[:2] != depth_map.shape:
        image_resized = image.resize((w, h), PILImage.LANCZOS)
        img_arr = np.array(image_resized).astype(np.float32)

    # Camera intrinsics from FOV
    focal = w / (2 * np.tan(np.radians(fov / 2)))

    # Generate 3D points
    yy, xx = np.mgrid[0:h, 0:w]
    z = depth_map * 10.0 + 0.1  # scale depth to reasonable range

    x = (xx - w / 2) * z / focal
    y = (yy - h / 2) * z / focal

    # Stack
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = img_arr.reshape(-1, 3) / 255.0

    # Filter out background (depth > 0.9)
    mask = depth_map.reshape(-1) < 0.9
    points = points[mask]
    colors = colors[mask]

    return np.hstack([points, colors])


def depth_to_mesh(depth_map: np.ndarray, image, fov: float = 60.0):
    """Convert depth map + RGB image to a trimesh.Trimesh.

    Uses Delaunay triangulation on the point cloud projected from depth.

    Args:
        depth_map: H x W normalized depth
        image: PIL.Image
        fov: Field of view

    Returns:
        trimesh.Trimesh
    """
    import trimesh

    pointcloud = depth_to_pointcloud(depth_map, image, fov)
    if len(pointcloud) < 3:
        logger.warning("Too few points for mesh generation")
        return trimesh.creation.box(extents=[1, 1, 1])

    points_3d = pointcloud[:, :3]
    colors_rgb = (pointcloud[:, 3:6] * 255).astype(np.uint8)

    # Subsample if too many points (for speed)
    max_points = 50000
    if len(points_3d) > max_points:
        indices = np.random.choice(len(points_3d), max_points, replace=False)
        points_3d = points_3d[indices]
        colors_rgb = colors_rgb[indices]

    # Try Delaunay triangulation on XY plane
    try:
        from scipy.spatial import Delaunay

        tri = Delaunay(points_3d[:, :2])
        faces = tri.simplices

        # Filter degenerate triangles (too large)
        edge_lengths = []
        for face in faces:
            pts = points_3d[face]
            edges = np.array([
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[2] - pts[1]),
                np.linalg.norm(pts[0] - pts[2]),
            ])
            edge_lengths.append(edges.max())
        edge_lengths = np.array(edge_lengths)
        threshold = np.percentile(edge_lengths, 95)
        valid = edge_lengths < threshold
        faces = faces[valid]

    except ImportError:
        logger.warning("scipy not available — using convex hull instead")
        cloud = trimesh.PointCloud(points_3d)
        return cloud.convex_hull

    # Build mesh
    alpha = np.full((len(colors_rgb), 1), 255, dtype=np.uint8)
    vertex_colors = np.hstack([colors_rgb, alpha])

    mesh = trimesh.Trimesh(
        vertices=points_3d,
        faces=faces,
        vertex_colors=vertex_colors,
    )

    logger.info(f"Depth-to-mesh: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")
    return mesh
