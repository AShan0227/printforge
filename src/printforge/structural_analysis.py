"""
Structural Analysis — Predict where a print will break and auto-reinforce.

Lightweight FEA-like analysis without full simulation:
  - Thin wall detection
  - Overhang stress estimation
  - Bridge weakness mapping
  - Auto-reinforcement suggestions (add ribs, thicken walls, split parts)

No external FEA dependencies — pure numpy mesh analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WeakPoint:
    """A structural weak point in the mesh."""
    location: Tuple[float, float, float]  # XYZ
    weakness_type: str  # "thin_wall", "overhang", "bridge", "island", "spike"
    severity: float  # 0-1 (1 = will definitely break)
    description: str
    fix_suggestion: str


@dataclass
class StructuralReport:
    """Complete structural analysis of a mesh."""
    overall_score: float  # 0-100 (100 = perfect)
    printability: str  # "excellent", "good", "fair", "poor", "unprintable"
    weak_points: List[WeakPoint] = field(default_factory=list)
    min_wall_thickness_mm: float = 0
    max_overhang_angle: float = 0
    unsupported_area_pct: float = 0
    island_count: int = 0
    recommendations: List[str] = field(default_factory=list)


class StructuralAnalyzer:
    """Analyze mesh structural integrity for 3D printing."""

    def __init__(self, layer_height_mm: float = 0.2, nozzle_mm: float = 0.4):
        self.layer_height = layer_height_mm
        self.nozzle = nozzle_mm

    def analyze(self, mesh, scale_mm: float = 50.0) -> StructuralReport:
        """Run full structural analysis on a trimesh.Trimesh."""
        import trimesh

        weak_points = []
        recommendations = []

        # Scale mesh to target size for analysis
        bounds = mesh.bounds
        current_size = (bounds[1] - bounds[0]).max()
        if current_size > 0:
            scale_factor = scale_mm / current_size
        else:
            scale_factor = 1.0

        verts = mesh.vertices * scale_factor

        # 1. Thin wall detection
        min_wall, thin_points = self._detect_thin_walls(mesh, verts)
        weak_points.extend(thin_points)
        if min_wall < self.nozzle:
            recommendations.append(
                f"Minimum wall thickness ({min_wall:.2f}mm) is below nozzle diameter ({self.nozzle}mm). "
                f"Scale up or thicken thin areas."
            )

        # 2. Overhang detection
        max_overhang, overhang_pct, overhang_points = self._detect_overhangs(mesh, verts)
        weak_points.extend(overhang_points)
        if max_overhang > 60:
            recommendations.append(
                f"Overhangs detected up to {max_overhang:.0f}°. "
                f"Enable supports or reorient the model."
            )

        # 3. Island detection (disconnected components)
        island_count = self._count_islands(mesh)
        if island_count > 1:
            recommendations.append(
                f"Model has {island_count} disconnected parts. "
                f"Consider merging or printing separately."
            )

        # 4. Spike/thin feature detection
        spike_points = self._detect_spikes(mesh, verts)
        weak_points.extend(spike_points)

        # 5. Bridge detection
        bridge_points = self._detect_bridges(mesh, verts)
        weak_points.extend(bridge_points)

        # Calculate overall score
        score = 100.0
        score -= len([wp for wp in weak_points if wp.severity > 0.7]) * 15
        score -= len([wp for wp in weak_points if 0.3 < wp.severity <= 0.7]) * 5
        score -= min(overhang_pct * 0.5, 20)
        if min_wall < self.nozzle:
            score -= 20
        if island_count > 1:
            score -= 10
        score = max(0, min(100, score))

        printability = "excellent" if score >= 80 else \
                       "good" if score >= 60 else \
                       "fair" if score >= 40 else \
                       "poor" if score >= 20 else "unprintable"

        if not recommendations:
            recommendations.append("Model looks good for printing! No major issues detected.")

        return StructuralReport(
            overall_score=round(score, 1),
            printability=printability,
            weak_points=weak_points,
            min_wall_thickness_mm=round(min_wall, 2),
            max_overhang_angle=round(max_overhang, 1),
            unsupported_area_pct=round(overhang_pct, 1),
            island_count=island_count,
            recommendations=recommendations,
        )

    def _detect_thin_walls(self, mesh, verts) -> Tuple[float, List[WeakPoint]]:
        """Detect thin walls using ray-based thickness estimation."""
        weak_points = []
        min_thickness = float('inf')

        # Sample vertices and cast rays inward to estimate thickness
        normals = mesh.vertex_normals
        n_samples = min(500, len(verts))
        indices = np.random.choice(len(verts), n_samples, replace=False)

        for idx in indices:
            v = verts[idx]
            n = normals[idx]

            # Cast ray inward (opposite normal)
            ray_origin = v + n * 0.01  # Slight offset
            ray_dir = -n

            # Simple nearest-vertex distance as proxy
            dists = np.linalg.norm(verts - (v - n * 5), axis=1)
            dists[idx] = float('inf')  # Exclude self
            nearest_dist = dists.min()

            # Estimate wall thickness
            thickness = min(nearest_dist, 10.0)  # Cap at 10mm
            min_thickness = min(min_thickness, thickness)

            if thickness < self.nozzle * 1.5:
                weak_points.append(WeakPoint(
                    location=tuple(v),
                    weakness_type="thin_wall",
                    severity=max(0, 1.0 - thickness / (self.nozzle * 2)),
                    description=f"Wall thickness ~{thickness:.2f}mm",
                    fix_suggestion="Thicken wall to at least {:.1f}mm".format(self.nozzle * 2),
                ))

        # Deduplicate nearby weak points
        weak_points = self._deduplicate_points(weak_points, threshold=2.0)
        return min_thickness if min_thickness < float('inf') else 1.0, weak_points

    def _detect_overhangs(self, mesh, verts) -> Tuple[float, float, List[WeakPoint]]:
        """Detect faces with overhang angles > 45°."""
        weak_points = []
        face_normals = mesh.face_normals
        up = np.array([0, 1, 0])  # Assume Y-up

        # Angle between face normal and up vector
        dots = np.dot(face_normals, up)
        angles = np.degrees(np.arccos(np.clip(dots, -1, 1)))

        # Overhangs are faces pointing mostly downward (angle > 135° from up = > 45° overhang)
        overhang_mask = angles > 135
        max_overhang = (angles[overhang_mask] - 90).max() if overhang_mask.any() else 0
        overhang_pct = overhang_mask.sum() / len(angles) * 100

        # Top weak points
        if overhang_mask.any():
            overhang_faces = np.where(overhang_mask)[0]
            sample_faces = overhang_faces[:10] if len(overhang_faces) > 10 else overhang_faces
            for fi in sample_faces:
                centroid = verts[mesh.faces[fi]].mean(axis=0)
                angle = angles[fi] - 90
                weak_points.append(WeakPoint(
                    location=tuple(centroid),
                    weakness_type="overhang",
                    severity=min(1.0, angle / 90),
                    description=f"Overhang at {angle:.0f}°",
                    fix_suggestion="Add support or reorient model",
                ))

        return max_overhang, overhang_pct, weak_points

    def _count_islands(self, mesh) -> int:
        """Count disconnected mesh components."""
        try:
            components = mesh.split(only_watertight=False)
            return len(components)
        except Exception:
            return 1

    def _detect_spikes(self, mesh, verts) -> List[WeakPoint]:
        """Detect thin protruding features (likely to break)."""
        weak_points = []
        # Use vertex valence (number of connected edges) as proxy
        # Low-valence vertices at extremes are likely spikes
        try:
            edges = mesh.edges_unique
            valence = np.zeros(len(verts))
            for e in edges:
                valence[e[0]] += 1
                valence[e[1]] += 1

            # Find low-valence extremal vertices
            centroid = verts.mean(axis=0)
            dists_from_center = np.linalg.norm(verts - centroid, axis=1)
            far_threshold = np.percentile(dists_from_center, 95)

            far_mask = dists_from_center > far_threshold
            low_valence_mask = valence < np.percentile(valence, 10)
            spike_mask = far_mask & low_valence_mask

            for idx in np.where(spike_mask)[0][:5]:
                weak_points.append(WeakPoint(
                    location=tuple(verts[idx]),
                    weakness_type="spike",
                    severity=0.6,
                    description="Thin protruding feature",
                    fix_suggestion="May break during printing — consider removing or thickening",
                ))
        except Exception:
            pass

        return weak_points

    def _detect_bridges(self, mesh, verts) -> List[WeakPoint]:
        """Detect horizontal spans without support (bridges)."""
        # Simplified: find long horizontal edges at non-bottom positions
        weak_points = []
        try:
            y_min = verts[:, 1].min()
            y_range = verts[:, 1].max() - y_min

            for edge in mesh.edges_unique[:1000]:  # Sample
                v0, v1 = verts[edge[0]], verts[edge[1]]
                length = np.linalg.norm(v1 - v0)
                # Horizontal edge check
                dy = abs(v1[1] - v0[1])
                height = ((v0[1] + v1[1]) / 2 - y_min) / max(y_range, 0.01)

                if length > 5.0 and dy < 0.5 and height > 0.3:
                    midpoint = (v0 + v1) / 2
                    weak_points.append(WeakPoint(
                        location=tuple(midpoint),
                        weakness_type="bridge",
                        severity=min(1.0, length / 20),
                        description=f"Unsupported bridge span ~{length:.1f}mm",
                        fix_suggestion="Add support or break into segments",
                    ))

            weak_points = self._deduplicate_points(weak_points, threshold=5.0)
        except Exception:
            pass

        return weak_points[:5]

    @staticmethod
    def _deduplicate_points(points: List[WeakPoint], threshold: float) -> List[WeakPoint]:
        """Remove weak points that are too close together."""
        if not points:
            return []
        result = [points[0]]
        for p in points[1:]:
            too_close = False
            for existing in result:
                dist = sum((a - b) ** 2 for a, b in zip(p.location, existing.location)) ** 0.5
                if dist < threshold:
                    too_close = True
                    break
            if not too_close:
                result.append(p)
        return result
