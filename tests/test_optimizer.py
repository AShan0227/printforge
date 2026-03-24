"""Tests for print optimizer and part splitter."""

import numpy as np
import pytest
import trimesh

from printforge.print_optimizer import (
    PrintOptimizer,
    Orientation,
    PrintEstimate,
    PrintabilityIssue,
    PRINTER_PRESETS,
    MATERIAL_DENSITY,
)
from printforge.part_splitter import (
    PartSplitter,
    SplitConfig,
    SplitResult,
    BuildVolume,
)


# ── Helpers ─────────────────────────────────────────────────────────

@pytest.fixture
def cube_mesh():
    """30mm watertight cube."""
    return trimesh.creation.box(extents=[30.0, 30.0, 30.0])


@pytest.fixture
def small_cube():
    """A 10mm cube — fits in any printer."""
    return trimesh.creation.box(extents=[10.0, 10.0, 10.0])


@pytest.fixture
def oversized_mesh():
    """A mesh that exceeds 256mm build volume on X axis."""
    return trimesh.creation.box(extents=[500.0, 100.0, 100.0])


@pytest.fixture
def tall_mesh():
    """Exceeds Z only."""
    return trimesh.creation.box(extents=[100.0, 100.0, 400.0])


@pytest.fixture
def sphere_mesh():
    """A sphere for overhang testing."""
    return trimesh.creation.icosphere(subdivisions=3, radius=25.0)


# ── PrintOptimizer tests ───────────────────────────────────────────

class TestPrintOptimizer:
    def test_find_best_orientation_returns_orientation(self, cube_mesh):
        optimizer = PrintOptimizer()
        result = optimizer.find_best_orientation(cube_mesh)
        assert isinstance(result, Orientation)
        assert result.height > 0
        assert result.base_area > 0
        assert result.rotation_matrix.shape == (4, 4)

    def test_orientation_prefers_lower_height(self):
        """A flat box should orient with the thin side up."""
        optimizer = PrintOptimizer()
        flat_box = trimesh.creation.box(extents=[100.0, 100.0, 10.0])
        result = optimizer.find_best_orientation(flat_box)
        # Best orientation should have height close to 10mm
        assert result.height <= 15.0

    def test_estimate_print_time(self, cube_mesh):
        optimizer = PrintOptimizer()
        time_min = optimizer.estimate_print_time(cube_mesh, layer_height=0.2)
        assert isinstance(time_min, float)
        assert time_min > 0
        # 30mm cube — allow wide range for simplified estimation
        assert 1 < time_min < 2000

    def test_estimate_print_time_increases_with_finer_layers(self, cube_mesh):
        optimizer = PrintOptimizer()
        time_02 = optimizer.estimate_print_time(cube_mesh, layer_height=0.2)
        time_01 = optimizer.estimate_print_time(cube_mesh, layer_height=0.1)
        assert time_01 > time_02  # finer layers = longer print

    def test_estimate_material(self, cube_mesh):
        optimizer = PrintOptimizer()
        est = optimizer.estimate_material(cube_mesh, infill=0.15, material="pla")
        assert isinstance(est, PrintEstimate)
        assert est.filament_grams > 0
        assert est.filament_meters > 0
        assert est.layer_count > 0
        assert est.print_time_minutes > 0

    def test_estimate_material_more_infill_uses_more(self, cube_mesh):
        optimizer = PrintOptimizer()
        est_15 = optimizer.estimate_material(cube_mesh, infill=0.15)
        est_50 = optimizer.estimate_material(cube_mesh, infill=0.50)
        assert est_50.filament_grams > est_15.filament_grams

    def test_check_printability_no_issues(self, small_cube):
        optimizer = PrintOptimizer()
        issues = optimizer.check_printability(small_cube, build_volume=(256, 256, 256))
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_check_printability_volume_exceeded(self, oversized_mesh):
        optimizer = PrintOptimizer()
        issues = optimizer.check_printability(oversized_mesh, build_volume=(256, 256, 256))
        volume_issues = [i for i in issues if i.category == "volume"]
        assert len(volume_issues) > 0
        assert volume_issues[0].severity == "error"

    def test_check_printability_not_watertight(self):
        """A mesh with removed faces should flag non-watertight."""
        mesh = trimesh.creation.box(extents=[30.0, 30.0, 30.0])
        # Remove some faces to break watertightness
        mesh.faces = mesh.faces[:-2]
        mesh.face_normals = mesh.face_normals[:-2] if len(mesh.face_normals) > 2 else mesh.face_normals

        optimizer = PrintOptimizer()
        issues = optimizer.check_printability(mesh)
        watertight_issues = [i for i in issues if i.category == "watertight"]
        assert len(watertight_issues) > 0

    def test_sphere_has_overhang_warning(self, sphere_mesh):
        optimizer = PrintOptimizer()
        issues = optimizer.check_printability(sphere_mesh)
        overhang_issues = [i for i in issues if i.category == "overhang"]
        assert len(overhang_issues) > 0

    def test_printer_presets_exist(self):
        assert "bambu-a1" in PRINTER_PRESETS
        assert "bambu-x1c" in PRINTER_PRESETS
        assert "prusa-mk4" in PRINTER_PRESETS
        for name, preset in PRINTER_PRESETS.items():
            assert "volume" in preset
            assert "name" in preset
            assert len(preset["volume"]) == 3

    def test_material_densities(self):
        assert "pla" in MATERIAL_DENSITY
        assert "petg" in MATERIAL_DENSITY
        for mat, density in MATERIAL_DENSITY.items():
            assert 0.5 < density < 2.0


# ── PartSplitter tests ─────────────────────────────────────────────

class TestPartSplitter:
    def test_no_split_needed(self, small_cube):
        splitter = PartSplitter()
        needs, axes = splitter.needs_splitting(small_cube)
        assert not needs
        assert axes == []

    def test_split_needed_detection(self, oversized_mesh):
        splitter = PartSplitter()
        needs, axes = splitter.needs_splitting(oversized_mesh)
        assert needs
        assert "x" in axes

    def test_split_no_split_returns_one_part(self, small_cube):
        splitter = PartSplitter()
        result = splitter.split(small_cube)
        assert isinstance(result, SplitResult)
        assert result.num_parts == 1
        assert result.fits_in_volume
        assert result.split_axes == []

    def test_split_produces_multiple_parts(self, oversized_mesh):
        splitter = PartSplitter()
        result = splitter.split(oversized_mesh)
        assert result.num_parts > 1
        assert len(result.parts) == result.num_parts

    def test_split_parts_have_correct_metadata(self, oversized_mesh):
        splitter = PartSplitter()
        result = splitter.split(oversized_mesh)
        for part in result.parts:
            assert part.mesh is not None
            assert len(part.mesh.faces) > 0
            assert part.part_index >= 0
            assert part.bounding_box is not None

    def test_split_tall_mesh_on_z(self, tall_mesh):
        splitter = PartSplitter()
        needs, axes = splitter.needs_splitting(tall_mesh)
        assert needs
        assert "z" in axes

        result = splitter.split(tall_mesh)
        assert result.num_parts >= 2

    def test_build_volume_from_string(self):
        vol = BuildVolume.from_string("180x180x180")
        assert vol.x == 180.0
        assert vol.y == 180.0
        assert vol.z == 180.0

    def test_build_volume_from_string_invalid(self):
        with pytest.raises(ValueError):
            BuildVolume.from_string("256x256")

    def test_custom_build_volume(self, cube_mesh):
        """A 30mm cube should need splitting with a 20mm build volume."""
        config = SplitConfig(build_volume=BuildVolume(20.0, 20.0, 20.0))
        splitter = PartSplitter(config)
        needs, axes = splitter.needs_splitting(cube_mesh)
        assert needs
        assert len(axes) == 3  # exceeds on all axes
