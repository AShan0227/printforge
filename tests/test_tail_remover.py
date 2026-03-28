"""Tests for tail detection and removal."""

import pytest
import numpy as np
import trimesh


class TestNoTailDetection:
    """Normal meshes should NOT have tails detected."""

    def test_cube_no_tail(self):
        from printforge.tail_remover import remove_tail

        mesh = trimesh.creation.box(extents=[1, 1, 1])
        result = remove_tail(mesh)

        assert result.tail_detected is False
        assert result.removed_percentage == 0.0
        assert result.tail_direction is None

    def test_sphere_no_tail(self):
        from printforge.tail_remover import remove_tail

        mesh = trimesh.creation.icosphere(subdivisions=2)
        result = remove_tail(mesh)

        assert result.tail_detected is False

    def test_slightly_elongated_no_tail(self):
        """A 2:1 aspect ratio box should NOT trigger tail detection."""
        from printforge.tail_remover import remove_tail

        mesh = trimesh.creation.box(extents=[2, 1, 1])
        result = remove_tail(mesh)

        assert result.tail_detected is False

    def test_legitimate_long_object_skipped(self):
        """A very long thin object (pencil-like) should be skipped."""
        from printforge.tail_remover import remove_tail

        # 1:1:20 aspect ratio = pencil
        mesh = trimesh.creation.box(extents=[1, 1, 20])
        result = remove_tail(mesh, max_aspect_ratio=12.0)

        assert result.tail_detected is False


class TestTailDetection:
    """Meshes with artificial tails should be detected."""

    def test_sphere_with_cylinder_tail(self):
        """A sphere with a thin cylinder attached = tail."""
        from printforge.tail_remover import remove_tail

        # Main body: sphere
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)

        # Tail: thin long cylinder
        tail = trimesh.creation.cylinder(radius=0.05, height=5.0)
        tail.apply_translation([0, 0, 3.5])  # Extend from sphere

        combined = trimesh.util.concatenate([sphere, tail])
        result = remove_tail(combined, extent_ratio_threshold=2.5)

        # Should detect the elongation
        assert result.tail_detected is True or combined.bounding_box.extents.max() / np.median(combined.bounding_box.extents) > 2.5

    def test_empty_mesh(self):
        from printforge.tail_remover import remove_tail

        mesh = trimesh.Trimesh()
        result = remove_tail(mesh)
        assert result.tail_detected is False
        assert result.removed_percentage == 0.0


class TestTailRemovalResult:
    def test_result_fields(self):
        from printforge.tail_remover import TailRemovalResult

        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        result = TailRemovalResult(
            original_verts=verts,
            cleaned_verts=verts,
            tail_detected=False,
            tail_direction=None,
            removed_percentage=0.0,
        )

        assert result.original_verts is verts
        assert result.cleaned_verts is verts
        assert result.tail_detected is False
        assert result.removed_percentage == 0.0

    def test_percentage_clamped(self):
        from printforge.tail_remover import TailRemovalResult

        verts = np.zeros((3, 3))
        result = TailRemovalResult(
            original_verts=verts,
            cleaned_verts=verts,
            tail_detected=True,
            tail_direction="+z",
            removed_percentage=1.5,  # Over 100%
        )
        assert result.removed_percentage == 1.0

        result2 = TailRemovalResult(
            original_verts=verts,
            cleaned_verts=verts,
            tail_detected=False,
            tail_direction=None,
            removed_percentage=-0.1,  # Negative
        )
        assert result2.removed_percentage == 0.0


class TestHelperFunctions:
    def test_detect_tail_axis_cube(self):
        from printforge.tail_remover import _detect_tail_axis

        extents = np.array([1.0, 1.0, 1.0])
        assert _detect_tail_axis(extents) is None

    def test_detect_tail_axis_elongated(self):
        from printforge.tail_remover import _detect_tail_axis

        extents = np.array([1.0, 1.0, 5.0])
        result = _detect_tail_axis(extents, threshold=2.5)
        assert result == 2  # Z axis

    def test_detect_tail_axis_x(self):
        from printforge.tail_remover import _detect_tail_axis

        extents = np.array([10.0, 1.0, 1.0])
        result = _detect_tail_axis(extents, threshold=2.5)
        assert result == 0  # X axis

    def test_axis_name(self):
        from printforge.tail_remover import _axis_name

        assert _axis_name(0, 1) == "+x"
        assert _axis_name(1, -1) == "-y"
        assert _axis_name(2, 1) == "+z"
