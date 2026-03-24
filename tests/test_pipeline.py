"""Tests for PrintForge pipeline, formats, and multi-view modules."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from printforge.pipeline import PrintForgePipeline, PipelineConfig, PipelineResult
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
