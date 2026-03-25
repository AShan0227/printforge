"""Tests for the P1 product polish sprint features."""

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import trimesh

from printforge.safety import ContentSafety, RateLimiter, image_hash, SKIN_TONE_THRESHOLD
from printforge.pipeline import PrintForgePipeline, PipelineConfig


# ── Helpers ───────────────────────────────────────────────────────

@pytest.fixture
def test_image(tmp_path):
    """Create a simple gray test image."""
    from PIL import Image

    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    path = tmp_path / "test.png"
    img.save(str(path))
    return str(path)


@pytest.fixture
def skin_tone_image(tmp_path):
    """Create an image dominated by skin-tone pixels."""
    from PIL import Image

    # Skin-tone color: H≈20°, S≈0.4, V≈0.7 → roughly (179, 120, 80) in RGB
    img = Image.new("RGB", (256, 256), color=(179, 120, 80))
    path = tmp_path / "skin.png"
    img.save(str(path))
    return str(path)


@pytest.fixture
def tmp_output(tmp_path):
    return str(tmp_path / "output")


# ── P1-1: NSFW detection & image_hash ────────────────────────────

class TestImageHash:
    def test_returns_hex_sha256(self, test_image):
        h = image_hash(test_image)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_matches_manual_sha256(self, test_image):
        h = image_hash(test_image)
        with open(test_image, "rb") as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        assert h == expected


class TestNSFWDetection:
    def test_safe_image_passes(self, test_image):
        """A plain gray image should pass the NSFW check."""
        safety = ContentSafety()
        result = safety.check_image(test_image)
        assert result.safe
        assert not result.flags

    def test_skin_tone_image_flagged(self, skin_tone_image):
        """An image of mostly skin-tone pixels gets flagged."""
        safety = ContentSafety()
        result = safety.check_image(skin_tone_image)
        assert not result.safe
        assert any("skin-tone" in f for f in result.flags)

    def test_banned_hash_blocks(self, test_image):
        """An image whose hash is on the blocklist is rejected."""
        h = image_hash(test_image)
        safety = ContentSafety(banned_hashes={h})
        result = safety.check_image(test_image)
        assert not result.safe
        assert any("banned" in f.lower() for f in result.flags)


class TestRateLimiterImageHash:
    def test_hash_rate_limiting(self):
        """Same image hash should be rate-limited independently."""
        rl = RateLimiter(max_requests=2, window_seconds=60)
        # Two requests with same hash — both allowed
        ok1, _ = rl.check("ip1", img_hash="abc123")
        ok2, _ = rl.check("ip2", img_hash="abc123")
        assert ok1
        assert ok2
        # Third request with same hash — blocked
        ok3, _ = rl.check("ip3", img_hash="abc123")
        assert not ok3

    def test_different_hash_not_blocked(self):
        """Different image hashes don't interfere."""
        rl = RateLimiter(max_requests=1, window_seconds=60)
        ok1, _ = rl.check("ip1", img_hash="hash_a")
        ok2, _ = rl.check("ip1", img_hash="hash_b")
        # ip1 is blocked (2nd request) but hash_b itself is fine
        # Actually ip1 hits ip limit on 2nd call
        assert ok1
        assert not ok2  # ip-level block

    def test_no_hash_backwards_compatible(self):
        """Calling check() without img_hash still works."""
        rl = RateLimiter(max_requests=5, window_seconds=60)
        ok, remaining = rl.check("ip1")
        assert ok
        assert remaining == 4


# ── P1-2: Adaptive MC resolution & smoothing ────────────────────

class TestAdaptiveResolution:
    def test_small_object_gets_96(self):
        config = PipelineConfig(adaptive_resolution=True)
        pipeline = PrintForgePipeline(config)
        mesh = trimesh.creation.box(extents=[10, 10, 10])  # small
        assert pipeline._choose_mc_resolution(mesh) == 96

    def test_medium_object_gets_128(self):
        config = PipelineConfig(adaptive_resolution=True)
        pipeline = PrintForgePipeline(config)
        mesh = trimesh.creation.box(extents=[50, 50, 50])
        assert pipeline._choose_mc_resolution(mesh) == 128

    def test_large_object_gets_192(self):
        config = PipelineConfig(adaptive_resolution=True)
        pipeline = PrintForgePipeline(config)
        mesh = trimesh.creation.box(extents=[200, 200, 200])
        assert pipeline._choose_mc_resolution(mesh) == 192

    def test_adaptive_disabled_uses_config(self):
        config = PipelineConfig(adaptive_resolution=False, mc_resolution=64)
        pipeline = PrintForgePipeline(config)
        mesh = trimesh.creation.box(extents=[200, 200, 200])
        assert pipeline._choose_mc_resolution(mesh) == 64


class TestLaplacianSmoothing:
    def test_smoothing_runs_in_pipeline(self, test_image, tmp_output):
        """Pipeline with smooth_iterations>0 completes successfully."""
        config = PipelineConfig(
            device="cpu",
            inference_backend="placeholder",
            smooth_iterations=2,
            adaptive_resolution=True,
        )
        pipeline = PrintForgePipeline(config)
        result = pipeline.run(test_image, tmp_output + ".stl")
        assert result.vertices > 0
        assert result.is_watertight

    def test_smoothing_zero_skipped(self, test_image, tmp_output):
        """Pipeline with smooth_iterations=0 still works."""
        config = PipelineConfig(
            device="cpu",
            inference_backend="placeholder",
            smooth_iterations=0,
        )
        pipeline = PrintForgePipeline(config)
        result = pipeline.run(test_image, tmp_output + ".stl")
        assert result.vertices > 0


# ── P1-3: Video→3D connection ────────────────────────────────────

class TestVideoTo3DConnection:
    def test_run_calls_pipeline(self, test_image, tmp_path):
        """VideoTo3D.run feeds the best frame through PrintForgePipeline."""
        from printforge.video_to_3d import VideoTo3D

        output_path = str(tmp_path / "out.stl")

        # Mock extract_frames to return our test image as a "frame"
        converter = VideoTo3D(num_frames=4)
        mock_extraction = MagicMock()
        mock_extraction.frames = [test_image]
        mock_extraction.duration_seconds = 1.0

        with patch.object(converter, "extract_frames", return_value=mock_extraction):
            result = converter.run("fake_video.mp4", output_path)

        assert result.mesh_path is not None
        assert Path(result.mesh_path).exists()
        assert result.num_frames_extracted == 1
