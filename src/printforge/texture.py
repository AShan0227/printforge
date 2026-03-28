"""
Texture Mapping — Apply colors and textures to 3D meshes from input images.
===========================================================================

Supports:
  - Color extraction (dominant + palette via K-means clustering)
  - Vertex coloring (projection-based and nearest-pixel)
  - Simple UV mapping (box projection)
  - Textured GLB export
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ColorPalette:
    """Extracted color palette from an image."""
    dominant_color: Tuple[int, int, int]
    palette: List[Tuple[int, int, int]]  # top N colors
    background_color: Tuple[int, int, int]


class TextureMapper:
    """Apply image-derived colors and textures to 3D meshes."""

    def __init__(self, palette_size: int = 5):
        self.palette_size = palette_size

    def extract_colors(self, image_path: str) -> ColorPalette:
        """Extract dominant colors from an image using histogram + K-means.

        Args:
            image_path: Path to the input image.

        Returns:
            ColorPalette with dominant color, palette, and detected background.
        """
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        img_small = img.resize((64, 64), Image.LANCZOS)
        pixels = np.array(img_small).reshape(-1, 3).astype(np.float32)

        # Detect background: sample corners
        arr = np.array(img).astype(np.float32)
        h, w = arr.shape[:2]
        corners = [
            arr[0, 0], arr[0, w - 1], arr[h - 1, 0], arr[h - 1, w - 1],
            arr[0, w // 2], arr[h - 1, w // 2],
        ]
        bg_color = tuple(int(c) for c in np.median(corners, axis=0).astype(int))

        # Simple K-means clustering
        palette = self._kmeans_colors(pixels, k=self.palette_size)

        # Dominant = largest cluster
        dominant = palette[0] if palette else bg_color

        return ColorPalette(
            dominant_color=dominant,
            palette=palette,
            background_color=bg_color,
        )

    def extract_colors_from_pil(self, image) -> ColorPalette:
        """Extract colors from a PIL Image directly."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            tmp_path = f.name
        try:
            return self.extract_colors(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def apply_vertex_colors(self, mesh, image, method: str = "projection"):
        """Apply colors from image to mesh vertices.

        Args:
            mesh: trimesh.Trimesh object
            image: PIL.Image (front view)
            method: "projection" or "nearest"

        Returns:
            trimesh.Trimesh with vertex_colors set
        """
        import trimesh
        from PIL import Image as PILImage

        img_arr = np.array(image.convert("RGB"))
        h, w = img_arr.shape[:2]

        verts = mesh.vertices.copy()
        # Normalize to [0, 1] range
        bounds = mesh.bounds
        mins, maxs = bounds[0], bounds[1]
        extents = maxs - mins
        extents[extents == 0] = 1.0  # avoid division by zero

        if method == "projection":
            colors = self._project_colors(verts, mins, extents, img_arr, h, w)
        elif method == "nearest":
            colors = self._nearest_colors(verts, mins, extents, img_arr, h, w)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Set vertex colors (RGBA)
        mesh.visual = trimesh.visual.ColorVisuals(
            mesh=mesh,
            vertex_colors=colors,
        )
        return mesh

    def apply_uv_mapping(self, mesh, image):
        """Apply box projection UV mapping with the image as texture.

        Args:
            mesh: trimesh.Trimesh object
            image: PIL.Image

        Returns:
            trimesh.Trimesh with texture applied
        """
        import trimesh

        # Box projection: project each face onto the most-aligned axis plane
        normals = mesh.face_normals
        verts = mesh.vertices
        faces = mesh.faces

        bounds = mesh.bounds
        mins, maxs = bounds[0], bounds[1]
        extents = maxs - mins
        extents[extents == 0] = 1.0

        uv_coords = np.zeros((len(verts), 2), dtype=np.float32)
        counts = np.zeros(len(verts), dtype=np.float32)

        for i, (face, normal) in enumerate(zip(faces, normals)):
            # Find dominant axis
            abs_normal = np.abs(normal)
            axis = np.argmax(abs_normal)

            for vi in face:
                v = (verts[vi] - mins) / extents  # normalized [0,1]
                if axis == 0:  # X-aligned → use Y, Z
                    uv = [v[1], v[2]]
                elif axis == 1:  # Y-aligned → use X, Z
                    uv = [v[0], v[2]]
                else:  # Z-aligned → use X, Y
                    uv = [v[0], v[1]]

                uv_coords[vi] += uv
                counts[vi] += 1

        # Average overlapping UV assignments
        mask = counts > 0
        uv_coords[mask] /= counts[mask, np.newaxis]

        # Create texture visual
        try:
            material = trimesh.visual.material.SimpleMaterial(image=image)
            tex_visual = trimesh.visual.TextureVisuals(
                uv=uv_coords,
                material=material,
            )
            mesh.visual = tex_visual
        except Exception as e:
            logger.warning(f"UV mapping failed, falling back to vertex colors: {e}")
            self.apply_vertex_colors(mesh, image)

        return mesh

    def export_textured_glb(self, mesh, output_path: str):
        """Export mesh with texture/colors as GLB.

        Args:
            mesh: trimesh.Trimesh with visual data
            output_path: Path for the output GLB file
        """
        mesh.export(output_path, file_type="glb")
        logger.info(f"Exported textured GLB to {output_path}")

    # ── Internal methods ──────────────────────────────────────────────

    @staticmethod
    def _project_colors(verts, mins, extents, img_arr, h, w):
        """Front-view projection: project vertices onto image plane."""
        normalized = (verts - mins) / extents  # [0, 1]

        # Map X → image column, Y → image row (inverted)
        cols = (normalized[:, 0] * (w - 1)).astype(int).clip(0, w - 1)
        rows = ((1 - normalized[:, 1]) * (h - 1)).astype(int).clip(0, h - 1)

        rgb = img_arr[rows, cols]  # (N, 3)
        alpha = np.full((len(verts), 1), 255, dtype=np.uint8)
        return np.hstack([rgb, alpha])

    @staticmethod
    def _nearest_colors(verts, mins, extents, img_arr, h, w):
        """Nearest pixel: each vertex gets color of closest image pixel."""
        normalized = (verts - mins) / extents
        cols = (normalized[:, 0] * (w - 1)).astype(int).clip(0, w - 1)
        rows = ((1 - normalized[:, 2]) * (h - 1)).astype(int).clip(0, h - 1)

        rgb = img_arr[rows, cols]
        alpha = np.full((len(verts), 1), 255, dtype=np.uint8)
        return np.hstack([rgb, alpha])

    @staticmethod
    def _kmeans_colors(pixels: np.ndarray, k: int = 5, max_iter: int = 20):
        """Simple K-means color clustering (no sklearn dependency).

        Args:
            pixels: (N, 3) float array of RGB values
            k: number of clusters
            max_iter: max iterations

        Returns:
            List of (R, G, B) tuples sorted by cluster size (largest first)
        """
        n = len(pixels)
        if n < k:
            return [tuple(int(c) for c in pixels[i]) for i in range(n)]

        # Initialize centroids with k random pixels
        rng = np.random.RandomState(42)
        indices = rng.choice(n, k, replace=False)
        centroids = pixels[indices].copy()

        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # Assign
            dists = np.linalg.norm(pixels[:, None] - centroids[None, :], axis=2)
            new_labels = np.argmin(dists, axis=1)

            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # Update centroids
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centroids[j] = pixels[mask].mean(axis=0)

        # Sort by cluster size
        sizes = [(labels == j).sum() for j in range(k)]
        order = np.argsort(sizes)[::-1]

        return [tuple(int(c) for c in centroids[j]) for j in order]
