"""
Tests for printforge.tail_remover
=================================
"""
import numpy as np
import pytest
import trimesh

from printforge.tail_remover import (
    remove_tail,
    TailRemovalResult,
    _detect_tail_axis,
    _connected_components,
    _axis_name,
    cleaned_mesh,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def sphere_mesh(radius: float = 0.5, center: tuple = (0, 0, 0), subdivisions: int = 2):
    """Return a UV sphere mesh."""
    m = trimesh.creation.uv_sphere(radius=radius, sections=subdivisions)
    m.apply_translation(center)
    return m


def cylinder_mesh(radius: float, height: float, center: tuple = (0, 0, 0)):
    """Return a cylinder aligned with the Z axis."""
    m = trimesh.creation.cylinder(radius=radius, height=height)
    m.apply_translation(center)
    return m


def sphere_plus_tail(
    radius: float = 0.5,
    tail_radius: float = 0.05,
    tail_length: float = 2.0,
    tail_center_z: float = None,
):
    """Return a sphere with a thin tail extending in the +Z direction."""
    if tail_center_z is None:
        tail_center_z = radius + tail_length / 2
    sphere = sphere_mesh(radius=radius, center=(0, 0, 0))
    tail = cylinder_mesh(
        radius=tail_radius,
        height=tail_length,
        center=(0, 0, tail_center_z),
    )
    return trimesh.util.concatenate([sphere, tail])


def pencil_like_mesh():
    """Return a long thin cylinder (pencil shape) — should NOT be trimmed."""
    body = cylinder_mesh(radius=0.05, height=2.0, center=(0, 0, 0))
    cone = trimesh.creation.cone(radius=0.05, height=0.2, sections=12)
    cone.apply_translation((0, 0, 1.0))
    return trimesh.util.concatenate([body, cone])


# --------------------------------------------------------------------------- #
# Test 1: Normal mesh (sphere) should NOT be flagged as having a tail
# --------------------------------------------------------------------------- #

class TestNormalMeshNoTail:

    @pytest.mark.parametrize("radius,subdivisions", [
        (0.5, 2), (1.0, 3), (0.3, 2),
    ])
    def test_sphere_no_tail(self, radius, subdivisions):
        mesh = sphere_mesh(radius=radius, subdivisions=subdivisions)
        result = remove_tail(mesh)

        assert isinstance(result, TailRemovalResult)
        assert result.tail_detected is False, "Sphere should not be detected as having a tail"
        assert result.tail_direction is None
        assert result.removed_percentage == 0.0
        np.testing.assert_array_almost_equal(result.original_verts, result.cleaned_verts)

    def test_box_no_tail(self):
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        result = remove_tail(mesh)
        assert result.tail_detected is False

    def test_torus_no_tail(self):
        mesh = trimesh.creation.torus()
        result = remove_tail(mesh)
        assert result.tail_detected is False

    def test_empty_mesh(self):
        """Empty mesh should not crash."""
        mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
        result = remove_tail(mesh)
        assert result.tail_detected is False


# --------------------------------------------------------------------------- #
# Test 2: Mesh with a tail should be detected and cleaned
# --------------------------------------------------------------------------- #

class TestTailDetectionAndRemoval:

    @pytest.mark.parametrize("tail_length", [1.5, 2.0, 3.0])
    def test_sphere_with_tail_detected(self, tail_length):
        mesh = sphere_plus_tail(radius=0.5, tail_radius=0.05, tail_length=tail_length)
        result = remove_tail(mesh)

        assert result.tail_detected is True, "Sphere+tail mesh should have tail detected"
        assert result.tail_direction in ("+z", "-z"), \
            f"Unexpected tail direction: {result.tail_direction}"
        assert result.removed_percentage > 0.05, \
            "Significant portion of vertices should be removed"
        assert result.removed_percentage < 0.95, \
            "Should not remove the entire mesh"

    def test_tail_direction_axis_idx_set(self):
        mesh = sphere_plus_tail(radius=0.5, tail_radius=0.05, tail_length=2.0)
        result = remove_tail(mesh)
        assert result.tail_axis_idx in (0, 1, 2)
        assert result.tail_sign in (1, -1)

    def test_removed_percentage_bounds(self):
        mesh = sphere_plus_tail(radius=0.5, tail_radius=0.05, tail_length=2.5)
        result = remove_tail(mesh)
        assert 0.0 <= result.removed_percentage <= 1.0

    def test_negative_direction_tail(self):
        """Tail extending in the -X direction."""
        sphere = sphere_mesh(radius=0.5, center=(0, 0, 0))
        tail = cylinder_mesh(
            radius=0.05,
            height=2.0,
            center=(-0.5 - 1.0, 0, 0),  # extends left in -X
        )
        mesh = trimesh.util.concatenate([sphere, tail])
        result = remove_tail(mesh)

        assert result.tail_detected is True
        assert result.tail_direction in ("+x", "-x"), \
            f"Expected X-axis tail, got {result.tail_direction}"

    def test_cleaned_mesh_is_valid(self):
        mesh = sphere_plus_tail(radius=0.5, tail_radius=0.05, tail_length=2.0)
        result = remove_tail(mesh)

        if result.tail_detected:
            cleaned = trimesh.Trimesh(vertices=result.cleaned_verts, faces=mesh.faces)
            assert len(cleaned.vertices) > 0
            assert len(cleaned.faces) > 0
            assert not np.any(np.isnan(cleaned.vertices))
            assert not np.any(np.isinf(cleaned.vertices))

    def test_cleaned_mesh_helper(self):
        """cleaned_mesh() helper should return correct vertices."""
        mesh = sphere_plus_tail(radius=0.5, tail_radius=0.05, tail_length=2.0)
        result = remove_tail(mesh)
        rebuilt = cleaned_mesh(result, mesh.faces)
        assert len(rebuilt.vertices) > 0


# --------------------------------------------------------------------------- #
# Test 3: Long thin meshes (pencil) should NOT be mistaken for tails
# --------------------------------------------------------------------------- #

class TestLegitimateLongMeshNotTrimmed:

    def test_pencil_not_trimmed(self):
        mesh = pencil_like_mesh()
        result = remove_tail(mesh)

        assert result.tail_detected is False, \
            "Legitimate elongated mesh (pencil) should not be trimmed"
        assert result.removed_percentage == 0.0

    def test_high_aspect_ratio_cylinder_not_trimmed(self):
        """A very long cylinder should be left alone."""
        mesh = cylinder_mesh(radius=0.05, height=5.0, center=(0, 0, 0))
        result = remove_tail(mesh)

        extents = mesh.bounding_box.extents
        aspect = extents.max() / max(extents.min(), 1e-9)
        assert aspect > 12, f"Test setup error: aspect ratio {aspect} not > 12"
        assert result.tail_detected is False

    def test_custom_max_aspect_ratio(self):
        """Even meshes above default threshold can be protected via max_aspect_ratio."""
        mesh = cylinder_mesh(radius=0.05, height=8.0, center=(0, 0, 0))
        result = remove_tail(mesh, max_aspect_ratio=50.0)
        assert result.tail_detected is False

    def test_custom_extent_ratio_threshold_higher(self):
        """Very high threshold should prevent tail detection."""
        mesh = sphere_plus_tail(radius=0.5, tail_radius=0.05, tail_length=2.0)
        result = remove_tail(mesh, extent_ratio_threshold=10.0)
        assert result.tail_detected is False

    def test_min_tail_vertices_ratio_prevents_overtrimming(self):
        """Tiny tail below ratio threshold should not be removed."""
        sphere = sphere_mesh(radius=0.5)
        tiny_tail = cylinder_mesh(radius=0.02, height=0.05, center=(0, 0, 0.55))
        mesh = trimesh.util.concatenate([sphere, tiny_tail])
        result = remove_tail(mesh, min_tail_vertices_ratio=0.05)
        assert result.tail_detected is False


# --------------------------------------------------------------------------- #
# Test: Helper functions
# --------------------------------------------------------------------------- #

class TestHelperFunctions:

    def test_detect_tail_axis_finds_elongated(self):
        extents = np.array([0.5, 0.5, 3.0])
        axis, sign = _detect_tail_axis(extents, threshold=2.5)
        assert axis == 2, "Z axis should be detected as elongated"

    def test_detect_tail_axis_returns_none_for_cube(self):
        extents = np.array([1.0, 1.0, 1.0])
        axis, sign = _detect_tail_axis(extents, threshold=2.5)
        assert axis is None

    def test_detect_tail_axis_partial(self):
        extents = np.array([1.0, 2.5, 1.0])  # Y is 2.5× the others
        axis, sign = _detect_tail_axis(extents, threshold=2.5)
        assert axis == 1

    def test_detect_tail_axis_respects_threshold(self):
        extents = np.array([1.0, 2.0, 1.0])  # Y is 2× the others — below threshold
        axis, sign = _detect_tail_axis(extents, threshold=2.5)
        assert axis is None

    def test_connected_components_sphere(self):
        mesh = sphere_mesh(radius=0.5)
        comps = _connected_components(mesh)
        assert len(comps) == 1
        assert len(comps[0]) == len(mesh.vertices)

    def test_connected_components_two_spheres_separate(self):
        s1 = sphere_mesh(radius=0.5, center=(0, 0, 0))
        s2 = sphere_mesh(radius=0.5, center=(10, 0, 0))  # far apart — separate
        mesh = trimesh.util.concatenate([s1, s2])
        comps = _connected_components(mesh)
        assert len(comps) >= 2

    def test_axis_name(self):
        assert _axis_name(0, 1) == "+x"
        assert _axis_name(0, -1) == "-x"
        assert _axis_name(2, 1) == "+z"
        assert _axis_name(1, -1) == "-y"


# --------------------------------------------------------------------------- #
# Test: PipelineResult integration
# --------------------------------------------------------------------------- #

class TestPipelineIntegration:

    def test_removed_percentage_never_negative(self):
        mesh = sphere_mesh(radius=0.5)
        result = remove_tail(mesh)
        assert result.removed_percentage >= 0.0

    def test_result_dataclass_fields(self):
        mesh = sphere_mesh(radius=0.5)
        result = remove_tail(mesh)
        assert hasattr(result, "original_verts")
        assert hasattr(result, "cleaned_verts")
        assert hasattr(result, "tail_detected")
        assert hasattr(result, "tail_direction")
        assert hasattr(result, "removed_percentage")
        assert hasattr(result, "tail_axis_idx")
        assert hasattr(result, "tail_sign")
