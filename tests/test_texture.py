"""Tests for texture mapping module."""

import pytest
import numpy as np
import tempfile
from pathlib import Path


class TestColorExtraction:
    def test_extract_from_solid_color(self):
        from PIL import Image
        from printforge.texture import TextureMapper

        # Create a solid red image
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            path = f.name

        mapper = TextureMapper(palette_size=3)
        palette = mapper.extract_colors(path)

        assert palette.dominant_color[0] > 200  # Red channel high
        assert palette.dominant_color[1] < 50   # Green channel low
        assert len(palette.palette) <= 3
        Path(path).unlink()

    def test_extract_from_pil(self):
        from PIL import Image
        from printforge.texture import TextureMapper

        img = Image.new("RGB", (64, 64), color=(0, 0, 255))
        mapper = TextureMapper()
        palette = mapper.extract_colors_from_pil(img)

        assert palette.dominant_color[2] > 200  # Blue dominant


class TestVertexColors:
    def test_projection_colors(self):
        import trimesh
        from PIL import Image
        from printforge.texture import TextureMapper

        mesh = trimesh.creation.box(extents=[1, 1, 1])
        img = Image.new("RGB", (64, 64), color=(0, 255, 0))

        mapper = TextureMapper()
        colored = mapper.apply_vertex_colors(mesh, img, method="projection")

        assert colored.visual.vertex_colors is not None
        assert len(colored.visual.vertex_colors) == len(mesh.vertices)

    def test_nearest_colors(self):
        import trimesh
        from PIL import Image
        from printforge.texture import TextureMapper

        mesh = trimesh.creation.box(extents=[1, 1, 1])
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))

        mapper = TextureMapper()
        colored = mapper.apply_vertex_colors(mesh, img, method="nearest")
        assert colored.visual.vertex_colors is not None


class TestUVMapping:
    def test_uv_mapping_box(self):
        import trimesh
        from PIL import Image
        from printforge.texture import TextureMapper

        mesh = trimesh.creation.box(extents=[1, 1, 1])
        img = Image.new("RGB", (64, 64), color=(255, 0, 0))

        mapper = TextureMapper()
        textured = mapper.apply_uv_mapping(mesh, img)
        # Should have visual data (either texture or vertex colors)
        assert textured.visual is not None


class TestKMeans:
    def test_kmeans_distinct_colors(self):
        from printforge.texture import TextureMapper

        # Create pixels with 2 clear clusters (with slight noise for stability)
        rng = np.random.RandomState(42)
        red = np.clip(rng.normal(255, 5, (50, 3)).astype(np.float32), 0, 255)
        red[:, 1] = np.clip(rng.normal(0, 5, 50).astype(np.float32), 0, 255)
        red[:, 2] = np.clip(rng.normal(0, 5, 50).astype(np.float32), 0, 255)

        blue = np.clip(rng.normal(0, 5, (50, 3)).astype(np.float32), 0, 255)
        blue[:, 2] = np.clip(rng.normal(255, 5, 50).astype(np.float32), 0, 255)

        pixels = np.vstack([red, blue])

        result = TextureMapper._kmeans_colors(pixels, k=2)
        assert len(result) == 2
        # Both clusters should be found (one red-ish, one blue-ish)
        max_r = max(c[0] for c in result)
        max_b = max(c[2] for c in result)
        assert max_r > 150  # Should have a reddish cluster
        assert max_b > 150  # Should have a bluish cluster
