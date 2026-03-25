"""Tests for Sprint 5: printer profiles, quality baselines, example generation."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from printforge.printer_profiles import (
    PRINTER_DB,
    PrinterProfile,
    get_profile,
    list_profiles,
)
from printforge.quality_baseline import (
    QUALITY_BASELINE,
    BaselineInfo,
    get_baseline,
    list_object_types,
)


# ── PrinterProfile tests ──────────────────────────────────────────

class TestPrinterProfiles:
    def test_all_eight_printers_exist(self):
        expected = [
            "bambu_x1c", "bambu_p1s", "bambu_a1", "bambu_a1_mini",
            "prusa_mk4", "prusa_mini", "creality_ender3", "creality_k1",
        ]
        for key in expected:
            assert key in PRINTER_DB, f"Missing printer: {key}"

    def test_get_profile_returns_correct_type(self):
        profile = get_profile("bambu_x1c")
        assert isinstance(profile, PrinterProfile)
        assert profile.name == "Bambu Lab X1 Carbon"

    def test_get_profile_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown printer"):
            get_profile("nonexistent_printer")

    def test_list_profiles_returns_sorted(self):
        profiles = list_profiles()
        assert profiles == sorted(profiles)
        assert len(profiles) == 8

    def test_build_volumes_are_positive(self):
        for key, profile in PRINTER_DB.items():
            x, y, z = profile.build_volume
            assert x > 0 and y > 0 and z > 0, f"{key} has invalid build volume"

    def test_nozzle_sizes_are_valid(self):
        for key, profile in PRINTER_DB.items():
            assert len(profile.nozzle_sizes) > 0, f"{key} has no nozzle sizes"
            for size in profile.nozzle_sizes:
                assert 0.1 <= size <= 1.5, f"{key} nozzle {size}mm out of range"

    def test_default_layer_height_valid(self):
        for key, profile in PRINTER_DB.items():
            assert 0.05 <= profile.default_layer_height <= 0.5

    def test_default_infill_valid(self):
        for key, profile in PRINTER_DB.items():
            assert 0.0 < profile.default_infill <= 1.0

    def test_bambu_a1_mini_smaller_volume(self):
        a1 = get_profile("bambu_a1")
        a1_mini = get_profile("bambu_a1_mini")
        a1_vol = a1.build_volume[0] * a1.build_volume[1] * a1.build_volume[2]
        mini_vol = a1_mini.build_volume[0] * a1_mini.build_volume[1] * a1_mini.build_volume[2]
        assert mini_vol < a1_vol

    def test_creality_k1_fastest(self):
        k1 = get_profile("creality_k1")
        ender3 = get_profile("creality_ender3")
        assert k1.max_speed > ender3.max_speed


# ── QualityBaseline tests ─────────────────────────────────────────

class TestQualityBaseline:
    def test_all_five_types_exist(self):
        expected = [
            "simple_geometric", "organic_shape", "mechanical_part",
            "text_logo", "figurine",
        ]
        for key in expected:
            assert key in QUALITY_BASELINE

    def test_get_baseline_returns_correct_type(self):
        info = get_baseline("simple_geometric")
        assert isinstance(info, BaselineInfo)
        assert info.object_type == "simple_geometric"

    def test_get_baseline_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown object type"):
            get_baseline("alien_artifact")

    def test_score_ranges_are_valid(self):
        for key, info in QUALITY_BASELINE.items():
            lo, hi = info.expected_score_range
            assert 0 <= lo <= hi <= 100, f"{key} has invalid score range ({lo}, {hi})"

    def test_simple_geometric_scores_highest(self):
        simple = get_baseline("simple_geometric")
        figurine = get_baseline("figurine")
        assert simple.expected_score_range[0] > figurine.expected_score_range[0]

    def test_all_baselines_have_tips(self):
        for key, info in QUALITY_BASELINE.items():
            assert len(info.tips) > 0, f"{key} has no tips"
            assert len(info.common_issues) > 0, f"{key} has no common issues"

    def test_list_object_types(self):
        types = list_object_types()
        assert len(types) == 5
        assert types == sorted(types)


# ── Example Model Generation tests ────────────────────────────────

class TestExampleModels:
    def test_trimesh_can_create_cube(self):
        cube = trimesh.creation.box(extents=[50.0, 50.0, 50.0])
        assert cube.is_watertight
        assert len(cube.faces) > 0

    def test_trimesh_can_create_sphere(self):
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
        assert sphere.is_watertight
        assert len(sphere.faces) > 0

    def test_trimesh_can_create_cylinder(self):
        cylinder = trimesh.creation.cylinder(radius=15.0, height=30.0, sections=64)
        assert cylinder.is_watertight
        assert len(cylinder.faces) > 0

    def test_export_stl_roundtrip(self, tmp_path):
        cube = trimesh.creation.box(extents=[50.0, 50.0, 50.0])
        stl_path = str(tmp_path / "test_cube.stl")
        cube.export(stl_path, file_type="stl")
        loaded = trimesh.load(stl_path)
        assert abs(len(loaded.faces) - len(cube.faces)) <= 2  # allow minor diff
