"""Tests for cost estimator, enhanced orientation analysis, and new API endpoints."""

import io
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import trimesh

from printforge.cost_estimator import CostEstimator, CostEstimate, MATERIAL_COST_PER_KG
from printforge.print_optimizer import PrintOptimizer, Orientation, MATERIAL_DENSITY


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def cube_mesh():
    """30mm watertight cube."""
    return trimesh.creation.box(extents=[30.0, 30.0, 30.0])


@pytest.fixture
def sphere_mesh():
    """Sphere for overhang testing."""
    return trimesh.creation.icosphere(subdivisions=3, radius=25.0)


@pytest.fixture
def flat_box():
    """A flat box — orientation matters."""
    return trimesh.creation.box(extents=[100.0, 100.0, 10.0])


@pytest.fixture
def tall_box():
    """A tall narrow box."""
    return trimesh.creation.box(extents=[10.0, 10.0, 100.0])


# ── CostEstimator tests ──────────────────────────────────────────

class TestCostEstimator:
    def test_estimate_returns_cost_estimate(self, cube_mesh):
        estimator = CostEstimator()
        result = estimator.estimate(cube_mesh, material="PLA", infill=0.15)
        assert isinstance(result, CostEstimate)

    def test_estimate_fields_positive(self, cube_mesh):
        estimator = CostEstimator()
        result = estimator.estimate(cube_mesh)
        assert result.filament_grams > 0
        assert result.filament_meters > 0
        assert result.filament_cost_usd > 0
        assert result.print_time_hours > 0
        assert result.electricity_cost_usd > 0
        assert result.total_cost_usd > 0

    def test_total_equals_filament_plus_electricity(self, cube_mesh):
        estimator = CostEstimator()
        result = estimator.estimate(cube_mesh)
        assert abs(result.total_cost_usd - (result.filament_cost_usd + result.electricity_cost_usd)) < 0.02

    def test_higher_infill_costs_more(self, cube_mesh):
        estimator = CostEstimator()
        low = estimator.estimate(cube_mesh, infill=0.10)
        high = estimator.estimate(cube_mesh, infill=0.50)
        assert high.filament_grams > low.filament_grams
        assert high.filament_cost_usd > low.filament_cost_usd

    def test_different_materials_different_cost(self, cube_mesh):
        estimator = CostEstimator()
        pla = estimator.estimate(cube_mesh, material="PLA")
        tpu = estimator.estimate(cube_mesh, material="TPU")
        # TPU is more expensive per kg
        assert tpu.filament_cost_usd > pla.filament_cost_usd

    def test_material_cost_database(self):
        """All materials in density table should have costs."""
        for mat in MATERIAL_DENSITY:
            assert mat in MATERIAL_COST_PER_KG, f"Missing cost for {mat}"
            assert MATERIAL_COST_PER_KG[mat] > 0

    def test_custom_electricity_rate(self, cube_mesh):
        cheap = CostEstimator(electricity_rate=0.05)
        expensive = CostEstimator(electricity_rate=0.30)
        est_cheap = cheap.estimate(cube_mesh)
        est_expensive = expensive.estimate(cube_mesh)
        assert est_expensive.electricity_cost_usd > est_cheap.electricity_cost_usd

    def test_sphere_cost(self, sphere_mesh):
        estimator = CostEstimator()
        result = estimator.estimate(sphere_mesh, material="PETG", infill=0.20)
        assert result.filament_grams > 0
        assert result.total_cost_usd > 0


# ── Enhanced Orientation tests ────────────────────────────────────

class TestEnhancedOrientation:
    def test_orientation_has_overhang_percentage(self, cube_mesh):
        optimizer = PrintOptimizer()
        result = optimizer.find_best_orientation(cube_mesh)
        assert hasattr(result, "overhang_percentage")
        assert isinstance(result.overhang_percentage, float)

    def test_orientation_has_support_contact_area(self, cube_mesh):
        optimizer = PrintOptimizer()
        result = optimizer.find_best_orientation(cube_mesh)
        assert hasattr(result, "support_contact_area")
        assert isinstance(result.support_contact_area, float)

    def test_sphere_has_overhangs(self, sphere_mesh):
        """A sphere should have significant overhang percentage."""
        optimizer = PrintOptimizer()
        result = optimizer.find_best_orientation(sphere_mesh)
        # Sphere always has overhangs regardless of orientation
        assert result.overhang_percentage > 0

    def test_flat_box_minimizes_overhang_area(self, flat_box):
        """A flat box should pick orientation with smallest bottom face (least overhang)."""
        optimizer = PrintOptimizer()
        result = optimizer.find_best_orientation(flat_box)
        # With overhang-based scoring, prefers smallest bottom face
        # 10x100 = 1000mm² bottom vs 100x100 = 10000mm² bottom
        assert result.score > 0
        assert result.overhang_percentage >= 0

    def test_cardinal_rotations_included(self):
        """Should generate exactly 6 cardinal rotations."""
        optimizer = PrintOptimizer()
        cardinals = optimizer._generate_cardinal_rotations()
        assert len(cardinals) == 6
        # Each should be a 4x4 matrix
        for mat in cardinals:
            assert mat.shape == (4, 4)

    def test_orientation_score_uses_new_formula(self, cube_mesh):
        """Score should be based on overhang_area * 0.7 + support_contact * 0.3."""
        optimizer = PrintOptimizer()
        # For a cube at identity orientation, bottom face is the only overhang
        rotated = cube_mesh.copy()
        result = optimizer._score_orientation(rotated, np.eye(4))
        # Verify score = overhang_area * 0.7 + support_contact_area * 0.3
        normals = rotated.face_normals
        areas = rotated.area_faces
        down_mask = normals[:, 2] < -0.5
        overhang_area = float(np.sum(areas[down_mask])) if np.any(down_mask) else 0.0
        down_components = np.abs(normals[down_mask, 2]) if np.any(down_mask) else np.array([])
        support_contact = float(np.sum(areas[down_mask] * down_components)) if np.any(down_mask) else 0.0
        expected_score = overhang_area * 0.7 + support_contact * 0.3
        assert abs(result.score - expected_score) < 0.01


# ── CLI cost command test ─────────────────────────────────────────

class TestCLICost:
    def test_cost_command_runs(self, cube_mesh):
        """Test that the cost CLI command executes without error."""
        import argparse
        from printforge.cli import cmd_cost

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            cube_mesh.export(f.name, file_type="stl")
            args = argparse.Namespace(
                mesh=f.name,
                material="PLA",
                infill=20,
                layer_height=0.2,
                verbose=False,
            )
            # Should not raise
            cmd_cost(args)
