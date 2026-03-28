"""Tests for TRELLIS backend integration."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import numpy as np


class TestTrellisConfig:
    """Test TrellisConfig defaults."""

    def test_default_config(self):
        from printforge.trellis_backend import TrellisConfig

        config = TrellisConfig()
        assert config.space_id == "JeffreyXiang/TRELLIS"
        assert config.ss_guidance_strength == 7.5
        assert config.slat_guidance_strength == 3.0
        assert config.mesh_simplify == 0.95
        assert config.texture_size == 1024

    def test_custom_config(self):
        from printforge.trellis_backend import TrellisConfig

        config = TrellisConfig(
            ss_guidance_strength=5.0,
            slat_sampling_steps=20,
            texture_size=2048,
        )
        assert config.ss_guidance_strength == 5.0
        assert config.slat_sampling_steps == 20
        assert config.texture_size == 2048


class TestTrellisBackend:
    """Test TrellisBackend with mocked Gradio client."""

    def _make_mock_glb(self, path: str):
        """Create a minimal GLB file that trimesh can load."""
        import trimesh

        mesh = trimesh.creation.box(extents=[1, 1, 1])
        mesh.export(path, file_type="glb")
        return path

    @patch("printforge.trellis_backend.TrellisBackend._get_client")
    def test_generate_success(self, mock_get_client):
        """Test successful generation with mocked client."""
        from printforge.trellis_backend import TrellisBackend

        # Create a temp GLB for the mock to return
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            glb_path = f.name
        self._make_mock_glb(glb_path)

        # Mock the Gradio client
        mock_client = MagicMock()
        mock_client.predict.side_effect = [
            None,  # start_session
            "/tmp/preprocessed.png",  # preprocess_image
            42,  # get_seed
            {"video": "/tmp/preview.mp4"},  # image_to_3d
            (glb_path, glb_path),  # extract_glb
        ]
        mock_get_client.return_value = mock_client

        backend = TrellisBackend()
        # Create a test image
        from PIL import Image

        test_img = Image.new("RGB", (512, 512), color="red")

        result = backend.generate(test_img)

        assert result is not None
        assert result.vertices > 0
        assert result.faces > 0
        assert result.mesh is not None

        # Cleanup
        Path(glb_path).unlink(missing_ok=True)

    @patch("printforge.trellis_backend.TrellisBackend._get_client")
    def test_generate_failure(self, mock_get_client):
        """Test graceful failure handling."""
        from printforge.trellis_backend import TrellisBackend

        mock_client = MagicMock()
        mock_client.predict.side_effect = Exception("GPU quota exceeded")
        mock_get_client.return_value = mock_client

        backend = TrellisBackend()
        from PIL import Image

        test_img = Image.new("RGB", (512, 512), color="blue")
        result = backend.generate(test_img)

        assert result is None  # Should return None, not raise

    def test_is_available_offline(self):
        """Test is_available returns False when Space is unreachable."""
        from printforge.trellis_backend import TrellisBackend, TrellisConfig

        # Use a fake space ID that won't resolve
        config = TrellisConfig(space_id="nonexistent/fake-space-12345")
        backend = TrellisBackend(config=config)
        assert backend.is_available() is False


class TestPipelineTrellisIntegration:
    """Test that pipeline correctly uses TRELLIS as first backend."""

    def test_fallback_chain_includes_trellis(self):
        """Verify TRELLIS is in the fallback chain."""
        from printforge.pipeline import PrintForgePipeline, PipelineConfig

        config = PipelineConfig(inference_backend="auto")
        pipeline = PrintForgePipeline(config=config)

        # Verify _infer_trellis method exists
        assert hasattr(pipeline, "_infer_trellis")

    def test_trellis_backend_option(self):
        """Verify 'trellis' is a valid backend option."""
        from printforge.pipeline import PipelineConfig

        config = PipelineConfig(inference_backend="trellis")
        assert config.inference_backend == "trellis"

    @patch("printforge.pipeline.PrintForgePipeline._infer_trellis")
    @patch("printforge.pipeline.PrintForgePipeline._infer_hunyuan3d")
    def test_trellis_tried_before_hunyuan(self, mock_hunyuan, mock_trellis):
        """Verify TRELLIS is tried before Hunyuan3D in auto mode."""
        import trimesh
        from printforge.pipeline import PrintForgePipeline, PipelineConfig

        # TRELLIS succeeds — Hunyuan should NOT be called
        mock_trellis.return_value = trimesh.creation.box(extents=[1, 1, 1])
        mock_hunyuan.return_value = None

        config = PipelineConfig(inference_backend="auto")
        pipeline = PrintForgePipeline(config=config)

        from PIL import Image

        test_img = Image.new("RGB", (512, 512), color="green")
        result = pipeline._infer_3d(test_img)

        mock_trellis.assert_called_once()
        mock_hunyuan.assert_not_called()
        assert result is not None
