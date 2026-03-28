"""Tests for depth estimator module."""

import pytest
import numpy as np


class TestDepthEstimatorFallback:
    def test_gradient_fallback(self):
        from PIL import Image
        from printforge.depth_estimator import DepthEstimator

        # Create a test image with gradient (left=bright, right=dark)
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        for x in range(64):
            arr[:, x, :] = int(255 * (1 - x / 63))
        img = Image.fromarray(arr)

        estimator = DepthEstimator()
        # Force fallback by not having HF token
        depth = estimator._estimate_gradient(img)

        assert depth.shape == (64, 64)
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    def test_estimate_returns_result(self):
        from PIL import Image
        from printforge.depth_estimator import DepthEstimator

        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        estimator = DepthEstimator()
        result = estimator.estimate(img)

        assert result.depth_map.shape == (64, 64)
        assert result.min_depth >= 0.0
        assert result.max_depth <= 1.0
        assert result.foreground_mask.shape == (64, 64)
        assert isinstance(result.estimated_depth_ratio, float)


class TestDepthToPointCloud:
    def test_pointcloud_shape(self):
        from PIL import Image
        from printforge.depth_estimator import depth_to_pointcloud

        depth = np.random.rand(32, 32).astype(np.float32) * 0.5
        img = Image.new("RGB", (32, 32), color=(255, 0, 0))

        pc = depth_to_pointcloud(depth, img, fov=60.0)

        assert pc.ndim == 2
        assert pc.shape[1] == 6  # x, y, z, r, g, b
        assert len(pc) > 0  # should have some foreground points

    def test_background_filtered(self):
        from PIL import Image
        from printforge.depth_estimator import depth_to_pointcloud

        # All background (depth = 1.0)
        depth = np.ones((32, 32), dtype=np.float32)
        img = Image.new("RGB", (32, 32), color=(0, 0, 0))

        pc = depth_to_pointcloud(depth, img, fov=60.0)
        # Most/all points should be filtered (depth > 0.9)
        assert len(pc) < 32 * 32


class TestDepthToMesh:
    def test_returns_mesh(self):
        from PIL import Image
        from printforge.depth_estimator import depth_to_mesh

        depth = np.random.rand(32, 32).astype(np.float32) * 0.5
        img = Image.new("RGB", (32, 32), color=(100, 200, 50))

        mesh = depth_to_mesh(depth, img, fov=60.0)

        assert hasattr(mesh, "vertices")
        assert hasattr(mesh, "faces")
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_too_few_points(self):
        from PIL import Image
        from printforge.depth_estimator import depth_to_mesh

        # All background = no foreground points
        depth = np.ones((4, 4), dtype=np.float32)
        img = Image.new("RGB", (4, 4), color=(0, 0, 0))

        mesh = depth_to_mesh(depth, img, fov=60.0)
        # Should return a fallback box
        assert len(mesh.vertices) > 0


class TestConvenienceFunction:
    def test_estimate_depth_from_file(self):
        import tempfile
        from PIL import Image
        from printforge.depth_estimator import estimate_depth

        img = Image.new("RGB", (64, 64), color=(200, 100, 50))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            path = f.name

        result = estimate_depth(path)
        assert result.depth_map.shape == (64, 64)

        from pathlib import Path
        Path(path).unlink()
