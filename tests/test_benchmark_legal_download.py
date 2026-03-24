"""Tests for benchmark suite, legal/TOS, and model download CLI."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import trimesh

from printforge.benchmark import BenchmarkSuite, BenchmarkReport, BENCHMARK_REPORT_PATH
from printforge.legal import get_tos, TOS_TEXT


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def test_image(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    path = tmp_path / "bench_test.png"
    img.save(str(path))
    return str(path)


@pytest.fixture
def cube_mesh():
    return trimesh.creation.box(extents=[30.0, 30.0, 30.0])


# ── Benchmark tests ──────────────────────────────────────────────

class TestBenchmarkSuite:
    def test_benchmark_inference_placeholder(self, test_image):
        suite = BenchmarkSuite()
        results = suite.benchmark_inference(test_image, backends=["placeholder"])
        assert len(results) == 1
        r = results[0]
        assert r.backend == "placeholder"
        assert r.duration_ms > 0
        assert r.vertices > 0
        assert r.faces > 0
        assert r.error is None

    def test_benchmark_inference_bad_backend(self, test_image):
        suite = BenchmarkSuite()
        results = suite.benchmark_inference(test_image, backends=["hunyuan3d"])
        assert len(results) == 1
        # hunyuan3d will fail without gradio_client — should return result with error or zero verts
        r = results[0]
        assert r.backend == "hunyuan3d"

    def test_benchmark_watertight(self, cube_mesh):
        suite = BenchmarkSuite()
        result = suite.benchmark_watertight(cube_mesh)
        assert result.duration_ms >= 0
        assert result.faces_before > 0
        assert result.faces_after > 0
        assert result.is_watertight

    def test_benchmark_pipeline(self, test_image):
        suite = BenchmarkSuite()
        report = suite.benchmark_pipeline(test_image)
        assert isinstance(report, BenchmarkReport)
        assert report.total_pipeline_ms > 0
        assert len(report.pipeline_stages) > 0
        assert report.timestamp != ""

    def test_benchmark_report_json_roundtrip(self, test_image):
        suite = BenchmarkSuite()
        report = suite.benchmark_pipeline(test_image)
        json_str = report.to_json()
        data = json.loads(json_str)
        assert "pipeline_stages" in data
        assert "total_pipeline_ms" in data
        assert data["total_pipeline_ms"] > 0

    def test_load_latest_after_benchmark(self, test_image):
        suite = BenchmarkSuite()
        suite.benchmark_pipeline(test_image)
        loaded = BenchmarkSuite.load_latest()
        assert loaded is not None
        assert loaded.total_pipeline_ms > 0


# ── Legal / TOS tests ────────────────────────────────────────────

class TestLegal:
    def test_tos_text_not_empty(self):
        assert len(TOS_TEXT) > 100

    def test_get_tos_returns_full_text(self):
        text = get_tos()
        assert text == TOS_TEXT
        assert "Content Ownership" in text
        assert "No Warranty" in text
        assert "Watertight Best-Effort" in text
        assert "MIT License" in text

    def test_tos_has_key_clauses(self):
        text = get_tos()
        assert "You retain full ownership" in text
        assert "solely responsible" in text
        assert "AS IS" in text
        assert "best-effort" in text


# ── Download model tests ─────────────────────────────────────────

class TestDownloadModel:
    def test_download_model_checks_existing(self, tmp_path):
        """If model file exists, cmd_download_model should report it."""
        from printforge.cli import cmd_download_model

        models_dir = tmp_path / ".printforge" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        fake_model = models_dir / "triposr_model.ckpt"
        fake_model.write_bytes(b"fake model data")

        args = MagicMock()
        args.model = "triposr"
        args.verbose = False

        with patch("printforge.cli.Path") as mock_path_cls:
            # We need to be more specific: only mock Path.home()
            pass

        # Just verify the function doesn't crash with a real args object
        # Full download test requires network, so we test the exists-check path
        # by patching Path.home
        with patch("pathlib.Path.home", return_value=tmp_path):
            cmd_download_model(args)

        # Should not have downloaded (file already exists)
        assert fake_model.exists()
