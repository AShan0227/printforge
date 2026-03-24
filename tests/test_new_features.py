"""Tests for multi-view enhance, batch processing, quality scoring, and mesh repair."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import trimesh

from printforge.multi_view import MultiViewEnhancer, MultiViewConfig
from printforge.batch import BatchProcessor, BatchResult
from printforge.quality import QualityScorer, QualityReport
from printforge.repair import MeshRepair, RepairReport
from printforge.pipeline import PipelineConfig


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def cube_mesh():
    return trimesh.creation.box(extents=[30.0, 30.0, 30.0])


@pytest.fixture
def test_image(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    path = tmp_path / "test.png"
    img.save(str(path))
    return str(path)


@pytest.fixture
def test_image_dir(tmp_path):
    """Create a directory with multiple test images."""
    from PIL import Image
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(3):
        img = Image.new("RGB", (512, 512), color=(100 + i * 50, 100, 100))
        img.save(str(img_dir / f"photo_{i}.png"))
    return str(img_dir)


# ── Multi-view enhance tests ─────────────────────────────────────

class TestMultiViewEnhance:
    def test_enhance_returns_four_views(self, test_image):
        enhancer = MultiViewEnhancer()
        views = enhancer.enhance(test_image)
        assert set(views.keys()) == {"front", "back", "left", "right"}

    def test_enhance_views_are_correct_size(self, test_image):
        enhancer = MultiViewEnhancer(MultiViewConfig(image_size=256))
        views = enhancer.enhance(test_image)
        for name, img in views.items():
            assert img.size == (256, 256), f"{name} has wrong size"

    def test_enhance_back_is_mirrored(self, test_image):
        """Back view should be a horizontal flip of the front."""
        from PIL import ImageOps
        enhancer = MultiViewEnhancer()
        views = enhancer.enhance(test_image)
        expected_back = ImageOps.mirror(views["front"])
        # Compare pixel data
        assert list(views["back"].getdata()) == list(expected_back.getdata())

    def test_enhance_with_user_supplied_back(self, tmp_path):
        """When user provides a back image, it should be used instead of mirrored."""
        from PIL import Image
        front_path = str(tmp_path / "front.png")
        back_path = str(tmp_path / "back.png")

        Image.new("RGB", (512, 512), color=(255, 0, 0)).save(front_path)
        Image.new("RGB", (512, 512), color=(0, 0, 255)).save(back_path)

        enhancer = MultiViewEnhancer()
        views = enhancer.enhance(front_path, extra_views={"back": back_path})

        # Back should be blue (user-supplied), not a mirror of red front
        back_pixel = views["back"].getpixel((0, 0))
        assert back_pixel == (0, 0, 255)

    def test_save_views_dict(self, test_image, tmp_path):
        enhancer = MultiViewEnhancer()
        views = enhancer.enhance(test_image)
        paths = enhancer.save_views(views, str(tmp_path / "views"))
        assert len(paths) == 4
        for p in paths:
            assert Path(p).exists()


# ── Batch processing tests ────────────────────────────────────────

class TestBatchProcessor:
    def test_collect_images(self, test_image_dir):
        paths = BatchProcessor.collect_images(test_image_dir)
        assert len(paths) == 3
        for p in paths:
            assert p.endswith(".png")

    def test_collect_images_bad_dir(self):
        with pytest.raises(FileNotFoundError):
            BatchProcessor.collect_images("/nonexistent/dir")

    def test_batch_process(self, test_image_dir, tmp_path):
        paths = BatchProcessor.collect_images(test_image_dir)
        config = PipelineConfig(device="cpu", inference_backend="placeholder")
        processor = BatchProcessor(config=config, max_workers=2)

        output_dir = str(tmp_path / "output")
        result = processor.process(paths, output_dir, output_format="stl")

        assert isinstance(result, BatchResult)
        assert result.succeeded == 3
        assert result.failed == 0
        assert result.total_duration_ms > 0
        for item in result.items:
            assert item.success
            assert Path(item.output_path).exists()

    def test_batch_progress_callback(self, test_image_dir, tmp_path):
        paths = BatchProcessor.collect_images(test_image_dir)
        config = PipelineConfig(device="cpu", inference_backend="placeholder")
        processor = BatchProcessor(config=config, max_workers=1)

        progress_calls = []

        def on_progress(done, total, item):
            progress_calls.append((done, total, item.success))

        processor.process(paths, str(tmp_path / "out"), "stl", progress_callback=on_progress)
        assert len(progress_calls) == 3
        assert progress_calls[-1][0] == 3  # last call: done == total


# ── Quality scoring tests ─────────────────────────────────────────

class TestQualityScorer:
    def test_perfect_cube_scores_high(self, cube_mesh):
        scorer = QualityScorer()
        report = scorer.score(cube_mesh)
        assert isinstance(report, QualityReport)
        assert report.total_score >= 70
        assert report.is_watertight
        assert report.watertight_score == 30.0
        assert report.grade in ("A", "B")

    def test_score_range(self, cube_mesh):
        scorer = QualityScorer()
        report = scorer.score(cube_mesh)
        assert 0 <= report.total_score <= 100

    def test_non_watertight_loses_30_points(self):
        """A mesh with removed faces should lose watertight points."""
        mesh = trimesh.creation.box(extents=[30.0, 30.0, 30.0])
        # Remove a face to break watertight
        mesh.faces = mesh.faces[:-1]
        mesh.remove_unreferenced_vertices()

        scorer = QualityScorer()
        report = scorer.score(mesh)
        assert not report.is_watertight
        assert report.watertight_score == 0.0

    def test_high_aspect_ratio_penalized(self):
        """An extremely elongated mesh should score poorly on aspect ratio."""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 100.0])
        scorer = QualityScorer()
        report = scorer.score(mesh)
        assert report.aspect_ratio >= 10.0
        assert report.aspect_ratio_score < 5.0

    def test_grade_letters(self, cube_mesh):
        scorer = QualityScorer()
        report = scorer.score(cube_mesh)
        assert report.grade in ("A", "B", "C", "D", "F")


# ── Mesh repair tests ────────────────────────────────────────────

class TestMeshRepair:
    def test_repair_watertight_mesh_is_noop(self, cube_mesh):
        repairer = MeshRepair()
        repaired, report = repairer.repair(cube_mesh)
        assert isinstance(report, RepairReport)
        assert report.is_watertight_after
        assert report.was_watertight_before
        assert not report.used_voxel_remesh

    def test_repair_fixes_broken_mesh(self):
        """A mesh with removed faces should be repaired to watertight."""
        mesh = trimesh.creation.box(extents=[30.0, 30.0, 30.0])
        # Remove faces to break it
        mesh.faces = mesh.faces[:-2]
        mesh.remove_unreferenced_vertices()
        assert not mesh.is_watertight

        repairer = MeshRepair(voxel_resolution=64)
        repaired, report = repairer.repair(mesh)
        assert report.is_watertight_after
        assert len(report.repairs_performed) > 0

    def test_repair_preserves_approximate_size(self, cube_mesh):
        repairer = MeshRepair()
        repaired, report = repairer.repair(cube_mesh)
        orig_extents = cube_mesh.bounding_box.extents
        new_extents = repaired.bounding_box.extents
        for o, n in zip(orig_extents, new_extents):
            assert abs(o - n) < 5.0  # within 5mm tolerance

    def test_repair_report_fields(self, cube_mesh):
        repairer = MeshRepair()
        _, report = repairer.repair(cube_mesh)
        assert report.input_faces > 0
        assert report.output_faces > 0
        assert report.input_vertices > 0
        assert report.output_vertices > 0

    def test_repair_type_check(self):
        repairer = MeshRepair()
        with pytest.raises(TypeError):
            repairer.repair("not a mesh")
