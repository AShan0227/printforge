"""Quality baseline data: expected scores and guidance for different object types."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BaselineInfo:
    """Expected quality baseline for an object type."""
    object_type: str
    expected_score_range: Tuple[float, float]  # (min, max) out of 100
    common_issues: List[str]
    tips: List[str]


QUALITY_BASELINE: Dict[str, BaselineInfo] = {
    "simple_geometric": BaselineInfo(
        object_type="simple_geometric",
        expected_score_range=(85, 100),
        common_issues=[
            "Sharp edges may need chamfering for FDM",
            "Internal corners can collect stress",
        ],
        tips=[
            "Use 0.2mm layer height for good surface finish",
            "15% infill is sufficient for decorative prints",
            "No supports needed for cubes, cylinders, etc.",
        ],
    ),
    "organic_shape": BaselineInfo(
        object_type="organic_shape",
        expected_score_range=(60, 85),
        common_issues=[
            "Overhangs above 45 degrees require supports",
            "Thin features may not print reliably",
            "Surface quality varies with curvature",
        ],
        tips=[
            "Enable tree supports for complex overhangs",
            "Use 0.12mm layer height for smooth curves",
            "Consider resin printing for fine organic detail",
        ],
    ),
    "mechanical_part": BaselineInfo(
        object_type="mechanical_part",
        expected_score_range=(70, 95),
        common_issues=[
            "Tolerances may drift with FDM layer adhesion",
            "Threads and holes need compensation (0.2mm offset)",
            "Internal cavities may trap support material",
        ],
        tips=[
            "Print functional parts in PETG or ABS for strength",
            "Orient so load-bearing axis aligns with layer lines",
            "Use 40-60% infill for structural parts",
            "Test-print tolerance gauges before final print",
        ],
    ),
    "text_logo": BaselineInfo(
        object_type="text_logo",
        expected_score_range=(65, 90),
        common_issues=[
            "Small text (<5mm height) may not resolve on FDM",
            "Thin strokes can break during removal from bed",
            "Bridging between letters can sag",
        ],
        tips=[
            "Minimum 6mm text height for FDM readability",
            "Emboss rather than engrave for better results",
            "Use a brim for thin logo pieces",
            "Consider resin for sub-3mm text detail",
        ],
    ),
    "figurine": BaselineInfo(
        object_type="figurine",
        expected_score_range=(50, 80),
        common_issues=[
            "Thin limbs and fingers often fail to print",
            "Complex overhangs need extensive supports",
            "Surface detail is limited by nozzle diameter",
            "High overhang percentage is common",
        ],
        tips=[
            "Use 0.4mm nozzle with 0.12mm layers for detail",
            "Tree supports remove more cleanly than linear",
            "Consider splitting at natural joints and gluing",
            "Resin SLA gives best figurine detail under 100mm",
        ],
    ),
}


def get_baseline(object_type: str) -> BaselineInfo:
    """Get quality baseline info for an object type.

    Args:
        object_type: One of 'simple_geometric', 'organic_shape',
                     'mechanical_part', 'text_logo', 'figurine'.

    Returns:
        BaselineInfo with expected scores, issues, and tips.

    Raises:
        KeyError: If the object type is not recognized.
    """
    if object_type not in QUALITY_BASELINE:
        available = ", ".join(sorted(QUALITY_BASELINE.keys()))
        raise KeyError(f"Unknown object type '{object_type}'. Available: {available}")
    return QUALITY_BASELINE[object_type]


def list_object_types() -> List[str]:
    """Return all available object type keys."""
    return sorted(QUALITY_BASELINE.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# Professional 3D Quality Assessment — Industry-standard metrics
# ═══════════════════════════════════════════════════════════════════════════════

# Thresholds
GOOD_ASPECT_RATIO_THRESHOLD = 5          # Triangle aspect ratio above this is bad
GOOD_DEPTH_RATIO_MIN = 0.15             # Z-depth/width ratio below this = "flat slab"
MAX_FLOATING_COMPONENTS = 1             # More than 1 connected component = floating debris
MAX_NORMAL_DEVIATION_RATIO = 0.05       # >5% of faces with sharp normal discontinuity = bad
ELONGATION_THRESHOLD = 3.0              # max_extent / median_extent above this = tail artifact


@dataclass
class QualityReport:
    """Comprehensive quality assessment of a 3D mesh."""
    # Individual scores (0-100)
    aspect_ratio_score: float = 100.0
    depth_ratio_score: float = 100.0
    component_score: float = 100.0
    normal_consistency_score: float = 100.0
    elongation_score: float = 100.0
    manifold_score: float = 100.0
    symmetry_score: float = 100.0
    watertight_score: float = 100.0

    # Overall
    overall_score: float = 100.0
    grade: str = "A"
    issues: List[str] = field(default_factory=list)

    # Raw data
    num_components: int = 1
    bad_aspect_ratio_pct: float = 0.0
    depth_width_ratio: float = 1.0
    normal_deviation_pct: float = 0.0
    elongation_ratio: float = 1.0
    non_manifold_edges: int = 0

    def summary(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 1),
            "grade": self.grade,
            "issues": self.issues,
            "scores": {
                "aspect_ratio": round(self.aspect_ratio_score, 1),
                "depth_ratio": round(self.depth_ratio_score, 1),
                "components": round(self.component_score, 1),
                "normal_consistency": round(self.normal_consistency_score, 1),
                "elongation": round(self.elongation_score, 1),
                "manifold": round(self.manifold_score, 1),
                "symmetry": round(self.symmetry_score, 1),
                "watertight": round(self.watertight_score, 1),
            },
            "raw": {
                "num_components": self.num_components,
                "bad_aspect_ratio_pct": round(self.bad_aspect_ratio_pct, 2),
                "depth_width_ratio": round(self.depth_width_ratio, 3),
                "normal_deviation_pct": round(self.normal_deviation_pct, 3),
                "elongation_ratio": round(self.elongation_ratio, 2),
                "non_manifold_edges": self.non_manifold_edges,
            },
        }


class MeshQualityAssessor:
    """Professional 3D mesh quality assessment with industry-standard metrics."""

    def assess(self, mesh) -> QualityReport:
        """Run all quality checks on a mesh.

        Args:
            mesh: trimesh.Trimesh

        Returns:
            QualityReport with per-metric scores and overall grade
        """
        import trimesh

        report = QualityReport()
        issues = []

        # 1. Aspect ratio analysis
        report.bad_aspect_ratio_pct, report.aspect_ratio_score = self._check_aspect_ratio(mesh)
        if report.bad_aspect_ratio_pct > 10:
            issues.append(f"High bad-aspect-ratio triangles: {report.bad_aspect_ratio_pct:.1f}%")

        # 2. Depth ratio (detect flat slabs)
        report.depth_width_ratio, report.depth_ratio_score = self._check_depth_ratio(mesh)
        if report.depth_width_ratio < GOOD_DEPTH_RATIO_MIN:
            issues.append(f"Flat slab detected: depth/width = {report.depth_width_ratio:.3f}")

        # 3. Connected components (floating debris)
        report.num_components, report.component_score = self._check_components(mesh)
        if report.num_components > MAX_FLOATING_COMPONENTS + 1:
            issues.append(f"Floating debris: {report.num_components} disconnected components")

        # 4. Normal consistency
        report.normal_deviation_pct, report.normal_consistency_score = self._check_normals(mesh)
        if report.normal_deviation_pct > MAX_NORMAL_DEVIATION_RATIO * 100:
            issues.append(f"Normal discontinuities: {report.normal_deviation_pct:.1f}% of faces")

        # 5. Elongation (tail artifact detection)
        report.elongation_ratio, report.elongation_score = self._check_elongation(mesh)
        if report.elongation_ratio > ELONGATION_THRESHOLD:
            issues.append(f"Possible tail artifact: elongation ratio {report.elongation_ratio:.1f}")

        # 6. Manifold check
        report.non_manifold_edges, report.manifold_score = self._check_manifold(mesh)
        if report.non_manifold_edges > 0:
            issues.append(f"Non-manifold edges: {report.non_manifold_edges}")

        # 7. Symmetry score
        report.symmetry_score = self._check_symmetry(mesh)

        # 8. Watertight
        report.watertight_score = 100.0 if mesh.is_watertight else 30.0
        if not mesh.is_watertight:
            issues.append("Mesh is not watertight")

        # Overall weighted score
        weights = {
            "aspect_ratio": 0.10,
            "depth_ratio": 0.15,
            "components": 0.15,
            "normals": 0.10,
            "elongation": 0.15,
            "manifold": 0.10,
            "symmetry": 0.05,
            "watertight": 0.20,
        }
        report.overall_score = (
            report.aspect_ratio_score * weights["aspect_ratio"]
            + report.depth_ratio_score * weights["depth_ratio"]
            + report.component_score * weights["components"]
            + report.normal_consistency_score * weights["normals"]
            + report.elongation_score * weights["elongation"]
            + report.manifold_score * weights["manifold"]
            + report.symmetry_score * weights["symmetry"]
            + report.watertight_score * weights["watertight"]
        )

        # Grade
        if report.overall_score >= 90:
            report.grade = "A"
        elif report.overall_score >= 75:
            report.grade = "B"
        elif report.overall_score >= 60:
            report.grade = "C"
        elif report.overall_score >= 40:
            report.grade = "D"
        else:
            report.grade = "F"

        report.issues = issues
        return report

    @staticmethod
    def _check_aspect_ratio(mesh) -> Tuple[float, float]:
        """Check triangle aspect ratios. Returns (bad_pct, score)."""
        triangles = mesh.triangles  # (N, 3, 3)
        edges = np.array([
            np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1),
            np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1),
            np.linalg.norm(triangles[:, 0] - triangles[:, 2], axis=1),
        ])  # (3, N)

        max_edges = edges.max(axis=0)
        min_edges = edges.min(axis=0)
        min_edges[min_edges == 0] = 1e-10

        ratios = max_edges / min_edges
        bad_pct = (ratios > GOOD_ASPECT_RATIO_THRESHOLD).sum() / len(ratios) * 100

        if bad_pct < 5:
            score = 100.0
        elif bad_pct < 15:
            score = 80.0
        elif bad_pct < 30:
            score = 50.0
        else:
            score = 20.0

        return float(bad_pct), score

    @staticmethod
    def _check_depth_ratio(mesh) -> Tuple[float, float]:
        """Check Z-depth vs width ratio. Returns (ratio, score)."""
        extents = mesh.bounding_box.extents
        # Sort: smallest = depth for flat objects
        sorted_ext = sorted(extents)
        min_ext = sorted_ext[0]
        max_ext = sorted_ext[-1]

        ratio = min_ext / max(max_ext, 1e-10)

        if ratio > 0.3:
            score = 100.0
        elif ratio > GOOD_DEPTH_RATIO_MIN:
            score = 70.0
        elif ratio > 0.05:
            score = 40.0
        else:
            score = 10.0

        return float(ratio), score

    @staticmethod
    def _check_components(mesh) -> Tuple[int, float]:
        """Count connected components. Returns (count, score)."""
        try:
            import trimesh
            splits = mesh.split(only_watertight=False)
            n = len(splits) if splits else 1
        except Exception:
            n = 1

        if n <= 1:
            score = 100.0
        elif n <= 2:
            score = 80.0
        elif n <= 5:
            score = 50.0
        else:
            score = 20.0

        return n, score

    @staticmethod
    def _check_normals(mesh) -> Tuple[float, float]:
        """Check normal consistency between adjacent faces. Returns (bad_pct, score)."""
        try:
            face_adjacency = mesh.face_adjacency
            normals = mesh.face_normals

            if len(face_adjacency) == 0:
                return 0.0, 100.0

            n0 = normals[face_adjacency[:, 0]]
            n1 = normals[face_adjacency[:, 1]]
            dots = np.sum(n0 * n1, axis=1)
            # Angle > 90° means sharp discontinuity
            bad = (dots < 0).sum()
            bad_pct = bad / len(face_adjacency) * 100

        except Exception:
            return 0.0, 100.0

        if bad_pct < 1:
            score = 100.0
        elif bad_pct < 5:
            score = 75.0
        elif bad_pct < 15:
            score = 45.0
        else:
            score = 15.0

        return float(bad_pct), score

    @staticmethod
    def _check_elongation(mesh) -> Tuple[float, float]:
        """Check for single-axis elongation (tail artifact). Returns (ratio, score)."""
        extents = mesh.bounding_box.extents
        median_ext = float(np.median(extents))
        max_ext = float(max(extents))

        ratio = max_ext / max(median_ext, 1e-10)

        if ratio < 2.0:
            score = 100.0
        elif ratio < ELONGATION_THRESHOLD:
            score = 70.0
        elif ratio < 5.0:
            score = 40.0
        else:
            score = 10.0

        return ratio, score

    @staticmethod
    def _check_manifold(mesh) -> Tuple[int, float]:
        """Check for non-manifold edges. Returns (count, score)."""
        try:
            # trimesh tracks edges shared by != 2 faces
            edges = mesh.edges_unique
            edge_face_count = np.zeros(len(edges))
            for fi, face in enumerate(mesh.faces):
                for i in range(3):
                    e = tuple(sorted([face[i], face[(i + 1) % 3]]))
                    # This is approximate — full check would use face_adjacency
            # Use a simpler proxy: watertight implies manifold
            if mesh.is_watertight:
                return 0, 100.0
            else:
                # Estimate from euler characteristic
                v, e, f = len(mesh.vertices), len(mesh.edges_unique), len(mesh.faces)
                euler = v - e + f
                non_manifold_est = max(0, abs(2 - euler))
                if non_manifold_est == 0:
                    return 0, 90.0
                elif non_manifold_est < 5:
                    return non_manifold_est, 70.0
                else:
                    return non_manifold_est, 30.0
        except Exception:
            return 0, 80.0

    @staticmethod
    def _check_symmetry(mesh) -> float:
        """Estimate bilateral symmetry (left-right). Returns score 0-100."""
        try:
            verts = mesh.vertices
            center = mesh.centroid

            # Mirror vertices across YZ plane (flip X)
            mirrored = verts.copy()
            mirrored[:, 0] = 2 * center[0] - mirrored[:, 0]

            # For each mirrored vertex, find nearest original vertex
            from scipy.spatial import cKDTree
            tree = cKDTree(verts)
            dists, _ = tree.query(mirrored, k=1)

            # Normalize by bounding box extent
            extent = mesh.bounding_box.extents.max()
            normalized_dists = dists / max(extent, 1e-10)

            # Mean normalized distance: 0 = perfect symmetry
            mean_dist = float(normalized_dists.mean())

            if mean_dist < 0.02:
                return 100.0
            elif mean_dist < 0.05:
                return 80.0
            elif mean_dist < 0.1:
                return 60.0
            elif mean_dist < 0.2:
                return 40.0
            else:
                return 20.0

        except ImportError:
            # scipy not available — skip symmetry check
            return 75.0  # neutral score
        except Exception:
            return 75.0


def assess_mesh_quality(mesh) -> dict:
    """Convenience function: assess mesh quality and return summary dict."""
    assessor = MeshQualityAssessor()
    report = assessor.assess(mesh)
    return report.summary()
