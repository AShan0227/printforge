"""Tests for PrintForge pipeline, formats, and multi-view modules."""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import trimesh

from printforge.pipeline import PrintForgePipeline, PipelineConfig, PipelineResult, HF_API_URL
from printforge.formats import export_3mf, export_stl, export_obj, SUPPORTED_FORMATS
from printforge.multi_view import MultiViewEnhancer, MultiViewConfig
from printforge.text_to_3d import TextTo3DPipeline, TextTo3DConfig


# ── Helpers ─────────────────────────────────────────────────────────

@pytest.fixture
def cube_mesh():
    """A simple watertight cube mesh."""
    return trimesh.creation.box(extents=[30.0, 30.0, 30.0])


@pytest.fixture
def tall_mesh():
    """A tall mesh that exceeds typical build volume on Z axis."""
    return trimesh.creation.box(extents=[50.0, 50.0, 400.0])


@pytest.fixture
def test_image(tmp_path):
    """Create a simple test image."""
    from PIL import Image

    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    path = tmp_path / "test.png"
    img.save(str(path))
    return str(path)


@pytest.fixture
def tmp_output(tmp_path):
    """Provide a temp output path."""
    return str(tmp_path / "output")


# ── Pipeline tests ──────────────────────────────────────────────────

class TestPipeline:
    def test_pipeline_produces_result(self, test_image, tmp_output):
        config = PipelineConfig(device="cpu")
        pipeline = PrintForgePipeline(config)
        result = pipeline.run(test_image, tmp_output + ".stl")

        assert isinstance(result, PipelineResult)
        assert result.vertices > 0
        assert result.faces > 0
        assert result.duration_ms > 0

    def test_placeholder_mesh_is_watertight(self):
        pipeline = PrintForgePipeline(PipelineConfig(device="cpu"))
        mesh = pipeline._create_placeholder_mesh()
        assert mesh.is_watertight

    def test_watertight_guarantee(self, test_image, tmp_output):
        config = PipelineConfig(device="cpu")
        pipeline = PrintForgePipeline(config)
        result = pipeline.run(test_image, tmp_output + ".stl")
        assert result.is_watertight

    def test_scaling(self, test_image, tmp_output):
        config = PipelineConfig(device="cpu", scale_mm=100.0, output_format="stl")
        pipeline = PrintForgePipeline(config)
        result = pipeline.run(test_image, tmp_output + ".stl")

        mesh = trimesh.load(result.mesh_path, force="mesh")
        max_extent = mesh.bounding_box.extents.max()
        assert abs(max_extent - 100.0) < 5.0  # within 5mm tolerance

    def test_output_formats(self, test_image, tmp_path):
        config = PipelineConfig(device="cpu")
        pipeline = PrintForgePipeline(config)

        for fmt in ("stl", "3mf"):
            out = str(tmp_path / f"test.{fmt}")
            config.output_format = fmt
            result = pipeline.run(test_image, out)
            assert Path(result.mesh_path).exists()


# ── Format export tests ─────────────────────────────────────────────

class TestFormats:
    def test_supported_formats_keys(self):
        assert "3mf" in SUPPORTED_FORMATS
        assert "stl" in SUPPORTED_FORMATS
        assert "obj" in SUPPORTED_FORMATS

    def test_export_stl(self, cube_mesh, tmp_path):
        path = export_stl(cube_mesh, str(tmp_path / "test.stl"))
        assert Path(path).exists()
        assert Path(path).suffix == ".stl"
        loaded = trimesh.load(path)
        assert len(loaded.faces) > 0

    def test_export_stl_ascii(self, cube_mesh, tmp_path):
        path = export_stl(cube_mesh, str(tmp_path / "test.stl"), binary=False)
        assert Path(path).exists()

    def test_export_obj(self, cube_mesh, tmp_path):
        path = export_obj(cube_mesh, str(tmp_path / "test.obj"))
        assert Path(path).exists()
        assert Path(path).suffix == ".obj"
        mtl_path = Path(path).with_suffix(".mtl")
        assert mtl_path.exists()

    def test_export_3mf(self, cube_mesh, tmp_path):
        path = export_3mf(cube_mesh, str(tmp_path / "test.3mf"))
        assert Path(path).exists()
        assert Path(path).suffix == ".3mf"

    def test_export_3mf_with_metadata(self, cube_mesh, tmp_path):
        meta = {"title": "Test Cube", "author": "PrintForge"}
        path = export_3mf(cube_mesh, str(tmp_path / "test.3mf"), metadata=meta)
        assert Path(path).exists()


# ── Multi-view tests ────────────────────────────────────────────────

class TestMultiView:
    def test_placeholder_returns_correct_count(self, test_image):
        enhancer = MultiViewEnhancer(MultiViewConfig(num_views=4))
        views = enhancer.generate_views(test_image)
        assert len(views) == 4

    def test_placeholder_returns_correct_size(self, test_image):
        enhancer = MultiViewEnhancer(MultiViewConfig(image_size=256))
        views = enhancer.generate_views(test_image)
        for v in views:
            assert v.size == (256, 256)

    def test_save_views(self, test_image, tmp_path):
        enhancer = MultiViewEnhancer(MultiViewConfig(num_views=3))
        views = enhancer.generate_views(test_image)
        paths = enhancer.save_views(views, str(tmp_path / "views"))
        assert len(paths) == 3
        for p in paths:
            assert Path(p).exists()


# ── Text-to-3D tests ───────────────────────────────────────────────

class TestTextTo3D:
    def test_prompt_generation(self):
        pipeline = TextTo3DPipeline()
        prompt = pipeline.generate_image_prompt("a cat-shaped vase")
        assert "cat-shaped vase" in prompt
        assert "white background" in prompt

    def test_custom_prompt_template(self):
        config = TextTo3DConfig(prompt_template="Create: {description}")
        pipeline = TextTo3DPipeline(config)
        prompt = pipeline.generate_image_prompt("a lamp")
        assert prompt == "Create: a lamp"

    def test_run_with_provided_image(self, test_image, tmp_output):
        """Test text-to-3D with a user-provided image (skipping generation)."""
        pipeline = TextTo3DPipeline()
        result = pipeline.run(
            description="a simple cube",
            output_path=tmp_output + ".stl",
            image_path=test_image,
            pipeline_config=PipelineConfig(device="cpu"),
        )
        assert result.mesh_path is not None
        assert result.pipeline_result is not None
        assert not result.used_fallback


# ── HuggingFace API backend tests ─────────────────────────────────

class TestHFAPIBackend:
    def test_hf_api_returns_mesh_on_success(self, test_image):
        """Mock the HTTP call and verify the API backend parses GLB."""
        from PIL import Image

        pipeline = PrintForgePipeline(PipelineConfig(device="cpu", inference_backend="api"))
        image = Image.open(test_image)

        # Create a real GLB from a trimesh cube
        cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        glb_bytes = cube.export(file_type="glb")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = glb_bytes
        mock_resp.raise_for_status = MagicMock()

        with patch.dict("os.environ", {"HF_TOKEN": "test-token"}), \
             patch("printforge.pipeline.requests.post", return_value=mock_resp) as mock_post:
            mesh = pipeline._infer_hf_api(image)

        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "Bearer test-token" in str(call_kwargs)

    def test_hf_api_skipped_without_token(self, test_image):
        """Without HF_TOKEN, _infer_hf_api returns None."""
        from PIL import Image

        pipeline = PrintForgePipeline(PipelineConfig(device="cpu"))
        image = Image.open(test_image)

        with patch.dict("os.environ", {}, clear=True):
            # Remove HF_TOKEN if present
            import os
            os.environ.pop("HF_TOKEN", None)
            mesh = pipeline._infer_hf_api(image)

        assert mesh is None

    def test_hf_api_failure_falls_back(self, test_image, tmp_output):
        """When API fails with backend='auto', pipeline falls back to placeholder."""
        pipeline = PrintForgePipeline(PipelineConfig(
            device="cpu", inference_backend="auto"
        ))

        with patch.dict("os.environ", {"HF_TOKEN": "test-token"}), \
             patch("printforge.pipeline.requests.post", side_effect=Exception("API down")):
            result = pipeline.run(test_image, tmp_output + ".stl")

        assert isinstance(result, PipelineResult)
        assert result.vertices > 0  # fell through to placeholder


# ── Background removal tests ──────────────────────────────────────

class TestBackgroundRemoval:
    def test_remove_background_with_rembg(self, test_image):
        """Mock rembg and verify background removal produces an RGB image."""
        from PIL import Image

        pipeline = PrintForgePipeline(PipelineConfig(
            device="cpu", remove_background=True
        ))
        image = Image.open(test_image).convert("RGB")

        # Create a mock RGBA output (white object on transparent)
        rgba_array = np.full((512, 512, 4), 255, dtype=np.uint8)
        rgba_array[:, :, 3] = 200  # partially transparent
        mock_rgba = Image.fromarray(rgba_array, "RGBA")

        with patch("printforge.pipeline.rembg_remove", create=True) as _:
            # Patch at the import site inside _remove_background
            import importlib
            mock_rembg_module = MagicMock()
            mock_rembg_module.remove = MagicMock(return_value=mock_rgba)

            with patch.dict("sys.modules", {"rembg": mock_rembg_module}):
                result = pipeline._remove_background(image)

        assert result.mode == "RGB"
        assert result.size == (512, 512)

    def test_remove_background_without_rembg(self, test_image):
        """Without rembg installed, returns original image unchanged."""
        from PIL import Image
        import sys

        pipeline = PrintForgePipeline(PipelineConfig(
            device="cpu", remove_background=True
        ))
        image = Image.open(test_image).convert("RGB")

        # Ensure rembg is not importable
        with patch.dict("sys.modules", {"rembg": None}):
            result = pipeline._remove_background(image)

        # Should return original image
        assert result.size == image.size

    def test_remove_background_disabled(self, test_image, tmp_output):
        """With remove_background=False, the step is skipped entirely."""
        config = PipelineConfig(device="cpu", remove_background=False)
        pipeline = PrintForgePipeline(config)
        result = pipeline.run(test_image, tmp_output + ".stl")

        assert isinstance(result, PipelineResult)
        assert "remove_background" not in result.stages


# ── Inference backend config tests ────────────────────────────────

class TestInferenceBackendConfig:
    def test_placeholder_backend(self, test_image, tmp_output):
        """Explicit placeholder backend produces result."""
        config = PipelineConfig(device="cpu", inference_backend="placeholder")
        pipeline = PrintForgePipeline(config)
        result = pipeline.run(test_image, tmp_output + ".stl")
        assert result.vertices > 0

    def test_progress_callback(self, test_image, tmp_output):
        """Progress callback is invoked during pipeline run."""
        stages_seen = []

        def on_progress(stage, pct):
            stages_seen.append((stage, pct))

        config = PipelineConfig(device="cpu", inference_backend="placeholder")
        pipeline = PrintForgePipeline(config)
        pipeline.run(test_image, tmp_output + ".stl", progress_callback=on_progress)

        stage_names = [s[0] for s in stages_seen]
        assert "inference" in stage_names
        assert "done" in stage_names
