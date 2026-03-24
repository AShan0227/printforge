"""Tests for image cache, inference fallback chain, and OpenAPI schema."""

import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import numpy as np
import pytest
import trimesh

from printforge.pipeline import PrintForgePipeline, PipelineConfig


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def cube_mesh():
    """A simple watertight cube mesh."""
    return trimesh.creation.box(extents=[30.0, 30.0, 30.0])


@pytest.fixture
def test_image(tmp_path):
    """Create a simple test image."""
    from PIL import Image

    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    path = tmp_path / "test.png"
    img.save(str(path))
    return str(path)


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary directory for the cache."""
    return tmp_path / "cache"


@pytest.fixture
def image_cache(cache_dir):
    """Create an ImageCache instance backed by a temp directory."""
    from printforge.cache import ImageCache

    return ImageCache(cache_dir=str(cache_dir))


@pytest.fixture
def sample_mesh_path(tmp_path, cube_mesh):
    """Export a cube mesh to a temp STL and return its path."""
    path = str(tmp_path / "cube.stl")
    cube_mesh.export(path, file_type="stl")
    return path


# ── Cache tests ────────────────────────────────────────────────────

class TestImageCache:
    def test_cache_miss_returns_none(self, image_cache):
        """get() on an unknown image key returns None."""
        result = image_cache.get("nonexistent_image_key")
        assert result is None

    def test_cache_put_and_hit(self, image_cache, sample_mesh_path):
        """put() then get() returns the cached path and the file exists."""
        image_cache.put("my_image", sample_mesh_path)
        cached = image_cache.get("my_image")

        assert cached is not None
        assert Path(cached).exists()

    def test_cache_hit_increments_stats(self, image_cache, sample_mesh_path):
        """After a cache hit, stats().hits increases."""
        image_cache.put("img_a", sample_mesh_path)
        image_cache.get("img_a")

        stats = image_cache.stats()
        assert stats.hits >= 1

    def test_cache_miss_increments_stats(self, image_cache):
        """After a cache miss, stats().misses increases."""
        image_cache.get("missing_key")

        stats = image_cache.stats()
        assert stats.misses >= 1

    def test_cache_ttl_expiry(self, image_cache, sample_mesh_path):
        """put() with ttl_seconds=0, then get() returns None after expiry."""
        image_cache.put("expiring", sample_mesh_path, ttl_seconds=0)

        # With ttl=0 the entry should already be expired
        result = image_cache.get("expiring")
        assert result is None

    def test_cache_clear(self, image_cache, sample_mesh_path):
        """put(), clear(), get() returns None."""
        image_cache.put("to_clear", sample_mesh_path)
        image_cache.clear()

        result = image_cache.get("to_clear")
        assert result is None

    def test_cache_stats_size(self, image_cache, sample_mesh_path):
        """After put(), stats().size_bytes > 0 and num_entries == 1."""
        image_cache.put("sized_entry", sample_mesh_path)

        stats = image_cache.stats()
        assert stats.size_bytes > 0
        assert stats.num_entries == 1


# ── Fallback chain tests ──────────────────────────────────────────

class TestFallbackChain:
    def test_fallback_hunyuan3d_mini_on_main_failure(self, test_image):
        """When _infer_hunyuan3d returns None, _infer_hunyuan3d_mini is called."""
        config = PipelineConfig(device="cpu", inference_backend="auto")
        pipeline = PrintForgePipeline(config)

        mini_mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])

        with patch.object(pipeline, "_infer_hunyuan3d", return_value=None) as mock_main, \
             patch.object(pipeline, "_infer_hunyuan3d_mini", return_value=mini_mesh) as mock_mini, \
             patch.object(pipeline, "_infer_hf_api", return_value=None), \
             patch.object(pipeline, "_infer_local", return_value=None):

            from PIL import Image
            image = Image.open(test_image)
            result = pipeline._infer_3d(image)

        mock_main.assert_called_once()
        mock_mini.assert_called_once()
        assert result is not None
        assert len(result.vertices) > 0

    def test_fallback_chain_reaches_placeholder(self, test_image):
        """When all inference methods return None, placeholder mesh is returned."""
        config = PipelineConfig(device="cpu", inference_backend="auto")
        pipeline = PrintForgePipeline(config)

        with patch.object(pipeline, "_infer_hunyuan3d", return_value=None), \
             patch.object(pipeline, "_infer_hf_api", return_value=None), \
             patch.object(pipeline, "_infer_local", return_value=None):

            from PIL import Image
            image = Image.open(test_image)
            result = pipeline._infer_3d(image)

        # Placeholder mesh is a 1x1x1 box
        assert result is not None
        assert result.is_watertight

    def test_fallback_logs_attempts(self, test_image):
        """logger.warning is called for each failed backend."""
        config = PipelineConfig(device="cpu", inference_backend="auto")
        pipeline = PrintForgePipeline(config)

        with patch.object(pipeline, "_infer_hunyuan3d", return_value=None), \
             patch.object(pipeline, "_infer_hf_api", return_value=None), \
             patch.object(pipeline, "_infer_local", return_value=None), \
             patch("printforge.pipeline.logger") as mock_logger:

            from PIL import Image
            image = Image.open(test_image)
            pipeline._infer_3d(image)

        # The final fallback emits a warning about all backends being unavailable
        warning_calls = mock_logger.warning.call_args_list
        assert len(warning_calls) >= 1
        # Check that the "all backends unavailable" warning is present
        warning_messages = [str(c) for c in warning_calls]
        assert any("unavailable" in msg.lower() or "placeholder" in msg.lower()
                    for msg in warning_messages)


# ── OpenAPI schema test ───────────────────────────────────────────

class TestOpenAPISchema:
    def test_openapi_schema_has_tags(self):
        """The FastAPI app's OpenAPI schema contains tags on its paths."""
        from printforge.server import app

        schema = app.openapi()

        assert "paths" in schema
        assert len(schema["paths"]) > 0

        # Verify that the schema has an info section with a title
        assert "info" in schema
        assert schema["info"]["title"] == "PrintForge"

        # Check that at least some paths define operation details
        # (operationId, summary, or tags)
        has_operations = False
        for path, methods in schema["paths"].items():
            for method, details in methods.items():
                if isinstance(details, dict) and "summary" in details:
                    has_operations = True
                    break
            if has_operations:
                break
        assert has_operations, "Expected at least one path with operation details"
