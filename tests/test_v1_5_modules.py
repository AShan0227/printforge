"""Tests for v1.5 modules: failure_predictor, competitor_monitor, analytics, safety, video_to_3d."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import trimesh

from printforge.failure_predictor import FailurePredictor, FailurePrediction, FailureRisk
from printforge.competitor_monitor import CompetitorMonitor, CompetitorUpdate
from printforge.analytics import Analytics
from printforge.safety import ContentSafety, RateLimiter, SafetyResult
from printforge.video_to_3d import VideoTo3D, FrameExtractionResult


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def cube_mesh():
    return trimesh.creation.box(extents=[30.0, 30.0, 30.0])


@pytest.fixture
def thin_mesh():
    """Mesh with a very thin dimension."""
    return trimesh.creation.box(extents=[0.2, 30.0, 30.0])


@pytest.fixture
def tall_mesh():
    """Mesh with extreme aspect ratio — topple risk."""
    return trimesh.creation.box(extents=[5.0, 5.0, 60.0])


@pytest.fixture
def test_image(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    path = tmp_path / "test.png"
    img.save(str(path))
    return str(path)


@pytest.fixture
def analytics_db(tmp_path):
    db_path = tmp_path / "test_analytics.db"
    return Analytics(db_path=db_path)


@pytest.fixture
def competitor_data_path(tmp_path):
    return tmp_path / "competitors.json"


# ── FailurePredictor tests ─────────────────────────────────────────

class TestFailurePredictor:
    def test_predict_returns_prediction(self, cube_mesh):
        predictor = FailurePredictor()
        result = predictor.predict(cube_mesh)
        assert isinstance(result, FailurePrediction)
        assert 0 <= result.risk_score <= 100

    def test_cube_not_critical(self, cube_mesh):
        """A simple cube should not be critical risk."""
        predictor = FailurePredictor()
        result = predictor.predict(cube_mesh)
        assert result.risk_level != "critical"
        assert result.risk_score < 100

    def test_thin_wall_detected(self, thin_mesh):
        predictor = FailurePredictor()
        result = predictor.predict(thin_mesh)
        risk_types = [r.type for r in result.risks]
        assert "thin_wall" in risk_types

    def test_tall_mesh_topple_risk(self, tall_mesh):
        predictor = FailurePredictor()
        result = predictor.predict(tall_mesh)
        risk_types = [r.type for r in result.risks]
        assert "topple_risk" in risk_types

    def test_small_feature_detected(self):
        tiny_mesh = trimesh.creation.box(extents=[0.1, 10.0, 10.0])
        predictor = FailurePredictor()
        result = predictor.predict(tiny_mesh)
        risk_types = [r.type for r in result.risks]
        assert "small_feature" in risk_types

    def test_predict_type_check(self):
        predictor = FailurePredictor()
        with pytest.raises(TypeError):
            predictor.predict("not a mesh")

    def test_risk_severity_values(self, thin_mesh):
        predictor = FailurePredictor()
        result = predictor.predict(thin_mesh)
        for risk in result.risks:
            assert risk.severity in ("low", "medium", "high", "critical")

    def test_disconnected_parts_detected(self):
        """Two separate cubes should trigger island detection."""
        mesh1 = trimesh.creation.box(extents=[10, 10, 10])
        mesh2 = trimesh.creation.box(extents=[10, 10, 10])
        mesh2.apply_translation([50, 0, 0])
        combined = trimesh.util.concatenate([mesh1, mesh2])
        predictor = FailurePredictor()
        result = predictor.predict(combined)
        risk_types = [r.type for r in result.risks]
        assert "islands" in risk_types


# ── CompetitorMonitor tests ──────────────────────────────────────

class TestCompetitorMonitor:
    def test_get_competitors(self, competitor_data_path):
        monitor = CompetitorMonitor(data_path=competitor_data_path)
        competitors = monitor.get_competitors()
        assert len(competitors) >= 4
        assert "meshy" in competitors
        assert "tripo" in competitors

    def test_check_updates_first_run(self, competitor_data_path):
        monitor = CompetitorMonitor(data_path=competitor_data_path)
        updates = monitor.check_updates()
        # First run: all competitors are "new"
        assert len(updates) >= 4
        assert all(isinstance(u, CompetitorUpdate) for u in updates)

    def test_check_updates_second_run_no_changes(self, competitor_data_path):
        monitor = CompetitorMonitor(data_path=competitor_data_path)
        monitor.check_updates()  # first run saves state
        monitor2 = CompetitorMonitor(data_path=competitor_data_path)
        updates = monitor2.check_updates()
        # Second run: no changes
        assert len(updates) == 0

    def test_get_summary(self, competitor_data_path):
        monitor = CompetitorMonitor(data_path=competitor_data_path)
        summary = monitor.get_summary()
        assert "competitors" in summary
        assert summary["total"] >= 4

    def test_data_saved_to_file(self, competitor_data_path):
        monitor = CompetitorMonitor(data_path=competitor_data_path)
        monitor.check_updates()
        assert competitor_data_path.exists()
        data = json.loads(competitor_data_path.read_text())
        assert "meshy" in data


# ── Analytics tests ──────────────────────────────────────────────

class TestAnalytics:
    def test_track_event(self, analytics_db):
        analytics_db.track("generation", format="3mf", duration_ms=1500.0)
        stats = analytics_db.get_stats()
        assert stats["total_events"] == 1

    def test_get_stats_empty(self, analytics_db):
        stats = analytics_db.get_stats()
        assert stats["total_events"] == 0
        assert stats["generations"] == 0
        assert stats["avg_duration_ms"] is None

    def test_track_multiple_events(self, analytics_db):
        analytics_db.track("generation", format="3mf", inference_backend="triposr", duration_ms=1000)
        analytics_db.track("generation", format="stl", inference_backend="hunyuan3d", duration_ms=2000)
        analytics_db.track("optimize", duration_ms=100)

        stats = analytics_db.get_stats()
        assert stats["total_events"] == 3
        assert stats["generations"] == 2
        assert stats["formats"]["3mf"] == 1
        assert stats["formats"]["stl"] == 1
        assert stats["backends"]["triposr"] == 1

    def test_avg_duration(self, analytics_db):
        analytics_db.track("generation", duration_ms=1000)
        analytics_db.track("generation", duration_ms=3000)
        stats = analytics_db.get_stats()
        assert stats["avg_duration_ms"] == 2000.0

    def test_clear(self, analytics_db):
        analytics_db.track("generation")
        analytics_db.clear()
        stats = analytics_db.get_stats()
        assert stats["total_events"] == 0

    def test_recent_events(self, analytics_db):
        for i in range(15):
            analytics_db.track(f"event_{i}")
        stats = analytics_db.get_stats()
        assert len(stats["recent_events"]) == 10  # capped at 10


# ── ContentSafety tests ──────────────────────────────────────────

class TestContentSafety:
    def test_valid_image(self, test_image):
        safety = ContentSafety()
        result = safety.check_image(test_image)
        assert isinstance(result, SafetyResult)
        assert result.safe
        assert len(result.flags) == 0

    def test_file_not_found(self):
        safety = ContentSafety()
        result = safety.check_image("/nonexistent/image.png")
        assert not result.safe
        assert "File not found" in result.flags

    def test_invalid_format(self, tmp_path):
        bad_file = tmp_path / "data.xyz"
        bad_file.write_text("not an image")
        safety = ContentSafety()
        result = safety.check_image(str(bad_file))
        assert not result.safe
        assert any("Invalid format" in f for f in result.flags)

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        safety = ContentSafety()
        result = safety.check_image(str(empty))
        assert not result.safe

    def test_oversized_file(self, tmp_path):
        """Simulate oversized file via stat mock."""
        from unittest.mock import PropertyMock
        big_file = tmp_path / "big.png"
        big_file.write_bytes(b"x" * 100)  # small file, but we'll mock size

        safety = ContentSafety()
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = 60 * 1024 * 1024  # 60MB
            result = safety.check_image(str(big_file))
        assert not result.safe
        assert any("too large" in f for f in result.flags)


# ── RateLimiter tests ────────────────────────────────────────────

class TestRateLimiter:
    def test_allows_under_limit(self):
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        allowed, remaining = limiter.check("127.0.0.1")
        assert allowed
        assert remaining == 4

    def test_blocks_over_limit(self):
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.check("127.0.0.1")
        allowed, remaining = limiter.check("127.0.0.1")
        assert not allowed
        assert remaining == 0

    def test_different_ips_independent(self):
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.check("1.1.1.1")
        limiter.check("1.1.1.1")
        allowed, _ = limiter.check("2.2.2.2")
        assert allowed  # different IP

    def test_get_usage(self):
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        limiter.check("10.0.0.1")
        limiter.check("10.0.0.1")
        usage = limiter.get_usage("10.0.0.1")
        assert usage["requests_used"] == 2
        assert usage["requests_remaining"] == 8
        assert usage["limit"] == 10


# ── VideoTo3D tests ──────────────────────────────────────────────

class TestVideoTo3D:
    def test_init_default_frames(self):
        converter = VideoTo3D()
        assert converter.num_frames == 8

    def test_init_custom_frames(self):
        converter = VideoTo3D(num_frames=16)
        assert converter.num_frames == 16

    def test_extract_file_not_found(self, tmp_path):
        converter = VideoTo3D()
        with pytest.raises(FileNotFoundError):
            converter.extract_frames("/nonexistent/video.mp4", str(tmp_path))

    def test_select_best_frames(self):
        converter = VideoTo3D()
        # Create synthetic frames with varying sharpness
        frames = []
        for i in range(10):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            if i == 5:
                # Make this frame extra sharp (high variance)
                frame[::2, ::2] = 255
                frame[1::2, 1::2] = 0
            frames.append((i, frame))

        best = converter._select_best_frames(frames, 4)
        assert len(best) == 4
        # Should maintain temporal order
        indices = [idx for idx, _ in best]
        assert indices == sorted(indices)
