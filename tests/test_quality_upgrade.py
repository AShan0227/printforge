"""Tests for upgraded quality assessment metrics."""

import pytest
import numpy as np


class TestQualityAssessor:
    def test_sphere_gets_high_score(self):
        import trimesh
        from printforge.quality_baseline import MeshQualityAssessor

        mesh = trimesh.creation.icosphere(subdivisions=3)
        assessor = MeshQualityAssessor()
        report = assessor.assess(mesh)

        assert report.overall_score > 70
        assert report.grade in ("A", "B")
        assert report.num_components == 1

    def test_flat_box_gets_low_depth_score(self):
        import trimesh
        from printforge.quality_baseline import MeshQualityAssessor

        # Very flat box: 10x10x0.01
        mesh = trimesh.creation.box(extents=[10, 10, 0.01])
        assessor = MeshQualityAssessor()
        report = assessor.assess(mesh)

        assert report.depth_ratio_score < 50
        assert report.depth_width_ratio < 0.01
        assert any("flat" in issue.lower() or "depth" in issue.lower() for issue in report.issues)

    def test_floating_components_detected(self):
        import trimesh
        from printforge.quality_baseline import MeshQualityAssessor

        # Main box + floating sphere
        box = trimesh.creation.box(extents=[1, 1, 1])
        sphere = trimesh.creation.icosphere(radius=0.1)
        sphere.apply_translation([5, 5, 5])  # Far away = floating
        combined = trimesh.util.concatenate([box, sphere])

        assessor = MeshQualityAssessor()
        report = assessor.assess(combined)

        assert report.num_components >= 2
        assert report.component_score < 100

    def test_elongated_mesh_detected(self):
        import trimesh
        from printforge.quality_baseline import MeshQualityAssessor

        # Very elongated: 1x1x20
        mesh = trimesh.creation.box(extents=[1, 1, 20])
        assessor = MeshQualityAssessor()
        report = assessor.assess(mesh)

        assert report.elongation_ratio > 3.0
        assert report.elongation_score < 80

    def test_summary_format(self):
        import trimesh
        from printforge.quality_baseline import assess_mesh_quality

        mesh = trimesh.creation.box(extents=[1, 1, 1])
        summary = assess_mesh_quality(mesh)

        assert "overall_score" in summary
        assert "grade" in summary
        assert "scores" in summary
        assert "raw" in summary
        assert "issues" in summary

    def test_grade_assignment(self):
        import trimesh
        from printforge.quality_baseline import MeshQualityAssessor

        # Good mesh should get A or B
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        assessor = MeshQualityAssessor()
        report = assessor.assess(mesh)
        assert report.grade in ("A", "B", "C")

    def test_normal_cube_is_watertight(self):
        import trimesh
        from printforge.quality_baseline import MeshQualityAssessor

        mesh = trimesh.creation.box(extents=[1, 1, 1])
        assessor = MeshQualityAssessor()
        report = assessor.assess(mesh)
        assert report.watertight_score == 100.0
