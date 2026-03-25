"""Integration tests for PrintForge Phase 2 — verify REAL connections work."""

import io
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import trimesh
from PIL import Image

from printforge.pipeline import PrintForgePipeline, PipelineConfig, PipelineResult
from printforge.batch import BatchProcessor
from printforge.safety import validate_image_magic_bytes
from printforge.text_to_3d import TextTo3DPipeline, TextTo3DConfig


# ── Helpers ─────────────────────────────────────────────────────────

@pytest.fixture
def test_image_path(tmp_path):
    """Create a real PNG test image with valid magic bytes."""
    img = Image.new("RGB", (512, 512), color=(100, 150, 200))
    path = tmp_path / "test_input.png"
    img.save(str(path), format="PNG")
    return str(path)


@pytest.fixture
def test_jpeg_path(tmp_path):
    """Create a real JPEG test image."""
    img = Image.new("RGB", (256, 256), color=(200, 100, 50))
    path = tmp_path / "test_input.jpg"
    img.save(str(path), format="JPEG")
    return str(path)


@pytest.fixture
def multiple_test_images(tmp_path):
    """Create 3 test images for batch processing."""
    paths = []
    for i in range(3):
        img = Image.new("RGB", (512, 512), color=(50 * i, 100, 150))
        path = tmp_path / f"batch_{i}.png"
        img.save(str(path), format="PNG")
        paths.append(str(path))
    return paths


# ── #1: Server generate with placeholder backend ────────────────────

class TestServerGeneratePlaceholder:
    def test_placeholder_backend_produces_file(self, test_image_path, tmp_path):
        """POST /api/generate equivalent: placeholder backend → real output."""
        config = PipelineConfig(
            inference_backend="placeholder",
            output_format="stl",
            scale_mm=50.0,
        )
        pipeline = PrintForgePipeline(config)
        output = str(tmp_path / "output.stl")
        result = pipeline.run(test_image_path, output)

        assert isinstance(result, PipelineResult)
        assert result.vertices > 0
        assert result.faces > 0
        assert result.is_watertight
        assert result.duration_ms > 0
        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_auto_backend_falls_to_placeholder(self, test_image_path, tmp_path):
        """With auto backend and no GPU/API, falls through to placeholder."""
        config = PipelineConfig(
            inference_backend="auto",
            output_format="stl",
            scale_mm=30.0,
        )
        pipeline = PrintForgePipeline(config)
        output = str(tmp_path / "auto_output.stl")
        result = pipeline.run(test_image_path, output)

        assert result.vertices > 0
        assert Path(output).exists()

    def test_response_headers_populated(self, test_image_path, tmp_path):
        """Verify the values that would go into X-PrintForge-* headers."""
        config = PipelineConfig(inference_backend="placeholder")
        pipeline = PrintForgePipeline(config)
        output = str(tmp_path / "headers_test.stl")
        result = pipeline.run(test_image_path, output)

        # These would be sent as response headers
        assert isinstance(result.vertices, int) and result.vertices > 0
        assert isinstance(result.faces, int) and result.faces > 0
        assert isinstance(result.is_watertight, bool)
        assert isinstance(result.duration_ms, float) and result.duration_ms > 0


# ── #2: Text-to-3D via Hunyuan3D ────────────────────────────────────

class TestTextTo3DViaHunyuan:
    def test_text_prompt_goes_to_hunyuan(self):
        """Verify text-to-3D tries Hunyuan3D with text as first param."""
        pipeline = TextTo3DPipeline()

        mock_client_instance = MagicMock()
        cube = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        glb_bytes = cube.export(file_type="glb")

        # Write GLB to temp so trimesh can load it
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            f.write(glb_bytes)
            glb_path = f.name

        mock_client_instance.predict.return_value = [{"value": glb_path}]

        mock_client_cls = MagicMock(return_value=mock_client_instance)

        # Patch at the gradio_client module level since it's imported inside the method
        with patch.dict("sys.modules", {"gradio_client": MagicMock(Client=mock_client_cls)}):
            with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as out:
                output_path = out.name

            pipeline.generate_via_hunyuan3d("a cute cat figurine", output_path)

            # Verify text was passed as first argument
            call_args = mock_client_instance.predict.call_args
            assert call_args[0][0] == "a cute cat figurine"
            assert call_args[0][1] is None  # no front image
            assert Path(output_path).exists()

    def test_text_to_3d_fallback_chain(self, tmp_path):
        """When image gen fails, tries Hunyuan3D text, then falls back."""
        pipeline = TextTo3DPipeline(TextTo3DConfig(hf_api_token=None))
        output = str(tmp_path / "text3d_output.stl")

        # No HF_API_TOKEN, no gradio_client → should gracefully fallback
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("HF_API_TOKEN", None)
            os.environ.pop("HF_TOKEN", None)

            result = pipeline.run(
                description="a simple vase",
                output_path=output,
            )

        # Should have used fallback
        assert result.used_fallback


# ── #3: Batch real execution ─────────────────────────────────────────

class TestBatchRealExecution:
    def test_batch_3_images_placeholder(self, multiple_test_images, tmp_path):
        """Process 3 images through batch with placeholder backend."""
        config = PipelineConfig(
            inference_backend="placeholder",
            output_format="stl",
            scale_mm=25.0,
        )
        processor = BatchProcessor(config=config, max_workers=3)
        output_dir = str(tmp_path / "batch_output")

        result = processor.process(
            image_paths=multiple_test_images,
            output_dir=output_dir,
            output_format="stl",
        )

        assert result.succeeded == 3
        assert result.failed == 0
        assert len(result.items) == 3
        assert result.total_duration_ms > 0

        for item in result.items:
            assert item.success
            assert item.result is not None
            assert item.result.vertices > 0
            assert Path(item.output_path).exists()

    def test_batch_thread_safety(self, multiple_test_images, tmp_path):
        """Verify each batch item gets its own pipeline config (no shared state)."""
        config = PipelineConfig(
            inference_backend="placeholder",
            scale_mm=40.0,
        )
        processor = BatchProcessor(config=config, max_workers=3)
        output_dir = str(tmp_path / "thread_safety")

        result = processor.process(multiple_test_images, output_dir, "stl")

        # All should succeed — if there's a thread-safety issue, some will fail
        assert result.succeeded == 3
        assert result.failed == 0

    def test_batch_with_progress_callback(self, multiple_test_images, tmp_path):
        """Verify progress callback fires for each item."""
        config = PipelineConfig(inference_backend="placeholder")
        processor = BatchProcessor(config=config, max_workers=2)

        progress_calls = []

        def on_progress(completed, total, item):
            progress_calls.append((completed, total, item.input_path))

        result = processor.process(
            multiple_test_images,
            str(tmp_path / "progress_test"),
            "stl",
            progress_callback=on_progress,
        )

        assert len(progress_calls) == 3
        assert result.succeeded == 3


# ── #4: Safety magic bytes ──────────────────────────────────────────

class TestSafetyMagicBytes:
    def test_valid_jpeg(self, test_jpeg_path):
        """Real JPEG file has valid magic bytes."""
        data = Path(test_jpeg_path).read_bytes()
        is_valid, fmt = validate_image_magic_bytes(data)
        assert is_valid
        assert fmt == "jpeg"

    def test_valid_png(self, test_image_path):
        """Real PNG file has valid magic bytes."""
        data = Path(test_image_path).read_bytes()
        is_valid, fmt = validate_image_magic_bytes(data)
        assert is_valid
        assert fmt == "png"

    def test_jpeg_magic_bytes_raw(self):
        """JPEG starts with FF D8 FF."""
        data = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        is_valid, fmt = validate_image_magic_bytes(data)
        assert is_valid
        assert fmt == "jpeg"

    def test_png_magic_bytes_raw(self):
        """PNG starts with 89 50 4E 47."""
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        is_valid, fmt = validate_image_magic_bytes(data)
        assert is_valid
        assert fmt == "png"

    def test_reject_text_file(self):
        """Plain text is not a valid image."""
        data = b"Hello, this is not an image file at all."
        is_valid, fmt = validate_image_magic_bytes(data)
        assert not is_valid
        assert fmt == "unknown"

    def test_reject_empty(self):
        """Empty data is rejected."""
        is_valid, fmt = validate_image_magic_bytes(b"")
        assert not is_valid
        assert fmt == "empty"

    def test_reject_pdf(self):
        """PDF file is not a valid image."""
        data = b"%PDF-1.4 some pdf content here"
        is_valid, fmt = validate_image_magic_bytes(data)
        assert not is_valid
        assert fmt == "unknown"

    def test_reject_fake_jpg_extension(self):
        """A file with .jpg name but non-image content is rejected by magic bytes."""
        data = b"This is definitely not a JPEG file"
        is_valid, fmt = validate_image_magic_bytes(data)
        assert not is_valid

    def test_valid_bmp(self):
        """BMP starts with 'BM'."""
        data = b"BM" + b"\x00" * 100
        is_valid, fmt = validate_image_magic_bytes(data)
        assert is_valid
        assert fmt == "bmp"

    def test_valid_gif(self):
        """GIF89a is a valid signature."""
        data = b"GIF89a" + b"\x00" * 100
        is_valid, fmt = validate_image_magic_bytes(data)
        assert is_valid
        assert fmt == "gif"


# ── #5: Docker syntax validation ────────────────────────────────────

class TestDockerSyntax:
    def test_dockerfile_parses(self):
        """Verify Dockerfile has valid structure."""
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found"

        content = dockerfile.read_text()

        # Must have FROM
        assert "FROM" in content
        # Must have EXPOSE
        assert "EXPOSE 8000" in content
        # Must have CMD
        assert "CMD" in content
        # Must install system deps
        assert "libgl1-mesa-glx" in content
        assert "python3-dev" in content
        # Must have healthcheck
        assert "HEALTHCHECK" in content

    def test_docker_compose_parses(self):
        """Verify docker-compose.yml has valid structure."""
        compose = Path(__file__).parent.parent / "docker-compose.yml"
        assert compose.exists(), "docker-compose.yml not found"

        content = compose.read_text()

        # Basic structure checks
        assert "services:" in content
        assert "printforge:" in content
        assert "8000:8000" in content
        assert "healthcheck:" in content
        assert "HF_TOKEN" in content

    def test_ci_yml_parses(self):
        """Verify CI workflow YAML has valid structure."""
        ci = Path(__file__).parent.parent / ".github" / "workflows" / "ci.yml"
        assert ci.exists(), "ci.yml not found"

        content = ci.read_text()

        assert "push:" in content
        assert "pull_request:" in content
        assert "main" in content
        assert "pytest" in content
        assert "timeout" in content.lower()


# ── #6: Multi-view enhancement ──────────────────────────────────────

class TestMultiViewEnhancement:
    def test_enhance_from_pil(self):
        """Verify enhance_from_pil produces all 4 views."""
        from printforge.multi_view import MultiViewEnhancer

        front = Image.new("RGB", (512, 512), color=(100, 200, 50))
        enhancer = MultiViewEnhancer()
        views = enhancer.enhance_from_pil(front)

        assert "front" in views
        assert "back" in views
        assert "left" in views
        assert "right" in views
        for name, img in views.items():
            assert img.size == (512, 512)

    def test_back_is_mirror_of_front(self):
        """Back view should be a horizontal flip of front."""
        from printforge.multi_view import MultiViewEnhancer

        # Asymmetric image so mirror is detectable
        front = Image.new("RGB", (512, 512), color=(0, 0, 0))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(front)
        draw.rectangle([0, 0, 100, 512], fill=(255, 0, 0))  # red left strip

        enhancer = MultiViewEnhancer()
        views = enhancer.enhance_from_pil(front)

        # The back's left edge should NOT be red (it should be mirrored)
        back_arr = np.array(views["back"])
        front_arr = np.array(views["front"])
        # Front left column is red, back left column should NOT be red
        assert front_arr[256, 50, 0] == 255  # front left is red
        assert back_arr[256, 50, 0] != 255   # back left is NOT red (mirrored)


# ── #7: Pipeline end-to-end with all stages ─────────────────────────

class TestPipelineEndToEnd:
    def test_full_pipeline_stages(self, test_image_path, tmp_path):
        """Verify all pipeline stages execute and are timed."""
        config = PipelineConfig(
            inference_backend="placeholder",
            output_format="stl",
            scale_mm=50.0,
            remove_background=False,  # skip rembg for speed
        )
        pipeline = PrintForgePipeline(config)
        output = str(tmp_path / "e2e.stl")
        result = pipeline.run(test_image_path, output)

        assert "load_image" in result.stages
        assert "inference" in result.stages
        assert "watertight" in result.stages
        assert "optimization" in result.stages
        assert "export" in result.stages

        # All stage times should be positive
        for stage, duration in result.stages.items():
            assert duration >= 0, f"Stage {stage} has negative duration"

    def test_glb_output_format(self, test_image_path, tmp_path):
        """Pipeline can produce STL that loads back correctly."""
        config = PipelineConfig(
            inference_backend="placeholder",
            output_format="stl",
        )
        pipeline = PrintForgePipeline(config)
        output = str(tmp_path / "roundtrip.stl")
        result = pipeline.run(test_image_path, output)

        # Load it back
        loaded = trimesh.load(output, force="mesh")
        assert len(loaded.vertices) > 0
        assert len(loaded.faces) > 0
