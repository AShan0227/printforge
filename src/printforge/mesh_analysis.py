"""
Mesh Analysis — Advanced mesh analysis for 3D printing.
=====================================================
- Volume & surface area computation (cm³, cm²)
- Overhang angle detection (>45°, >60°, >70°)
- Thin wall detection (< 2× nozzle width)
- Support material volume estimation
- Print difficulty score (1-10)
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────────────────
DEFAULT_NOZZLE_MM = 0.4
MIN_WALL_NOZZLE_WIDTHS = 2.0
SUPPORT_DENSITY = 0.35  # PLA g/cm³, sparse


# ── Dataclasses ─────────────────────────────────────────────────────────────────


@dataclass
class OverhangRegion:
    centroid_mm: tuple[float, float, float]
    area_mm2: float
    angle_deg: float
    height_above_plate_mm: float
    severity: str  # mild, moderate, severe


@dataclass
class ThinWallRegion:
    thickness_mm: float
    face_indices: list[int]
    severity: str  # critical, warning, info


@dataclass
class MeshAnalysisResult:
    volume_cm3: float
    surface_area_cm2: float
    bounding_box_mm: tuple[float, float, float]
    face_count: int
    vertex_count: int
    is_watertight: bool
    is_manifold: bool

    overhang_regions: list[OverhangRegion]
    total_overhang_area_mm2: float
    overhang_percentage: float
    severe_overhang_mm2: float

    thin_wall_regions: list[ThinWallRegion]
    min_wall_thickness_mm: float
    has_thin_walls: bool

    support_volume_cm3: float
    support_contact_area_cm2: float
    estimated_support_grams: float

    difficulty_score: float
    difficulty_factors: list[str]
    print_time_minutes: float
    filament_grams: float

    def summary(self) -> dict:
        return {
            "volume_cm3": round(self.volume_cm3, 4),
            "surface_area_cm2": round(self.surface_area_cm2, 4),
            "bounding_box_mm": [round(v, 2) for v in self.bounding_box_mm],
            "face_count": self.face_count,
            "vertex_count": self.vertex_count,
            "is_watertight": self.is_watertight,
            "is_manifold": self.is_manifold,
            "overhang": {
                "total_area_mm2": round(self.total_overhang_area_mm2, 2),
                "percentage": round(self.overhang_percentage, 2),
                "severe_area_mm2": round(self.severe_overhang_mm2, 2),
                "regions": [
                    {"centroid": list(r.centroid_mm), "area_mm2": round(r.area_mm2, 2),
                     "angle_deg": round(r.angle_deg, 1), "height_above_plate_mm": round(r.height_above_plate_mm, 2),
                     "severity": r.severity}
                    for r in self.overhang_regions[:20]  # limit for API
                ],
            },
            "thin_walls": {
                "has_thin_walls": self.has_thin_walls,
                "min_thickness_mm": round(self.min_wall_thickness_mm, 3),
                "regions": [
                    {"thickness_mm": round(r.thickness_mm, 3), "face_count": len(r.face_indices), "severity": r.severity}
                    for r in self.thin_wall_regions
                ],
            },
            "support": {
                "volume_cm3": round(self.support_volume_cm3, 4),
                "contact_area_cm2": round(self.support_contact_area_cm2, 4),
                "estimated_grams": round(self.estimated_support_grams, 2),
            },
            "print_difficulty": {
                "score": round(self.difficulty_score, 1),
                "factors": self.difficulty_factors,
                "print_time_minutes": round(self.print_time_minutes, 1),
                "filament_grams": round(self.filament_grams, 1),
            },
        }


# ── Main Analyzer ───────────────────────────────────────────────────────────────


class MeshAnalyzer:
    def __init__(
        self,
        nozzle_diameter_mm: float = DEFAULT_NOZZLE_MM,
        layer_height_mm: float = 0.2,
        infill: float = 0.15,
        material_density: float = 1.24,
    ):
        self.nozzle_diameter_mm = nozzle_diameter_mm
        self.layer_height_mm = layer_height_mm
        self.infill = infill
        self.material_density = material_density

    def analyze(self, mesh) -> MeshAnalysisResult:
        import trimesh

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh).__name__}")

        extents = mesh.bounding_box.extents

        try:
            volume_cm3 = float(mesh.volume) / 1000.0
        except Exception:
            volume_cm3 = 0.0

        surface_area_cm2 = float(mesh.area) / 100.0
        is_watertight = bool(mesh.is_watertight)

        # Overhang analysis
        overhang_regions, total_oh_area, oh_pct, severe_area = self._analyze_overhangs(mesh)

        # Thin wall
        thin_wall_regions, min_wall_mm = self._detect_thin_walls(mesh)

        # Support
        support_vol, support_contact, support_grams = self._estimate_support(overhang_regions)

        # Difficulty
        difficulty, factors, print_time, filament = self._compute_difficulty(
            mesh, volume_cm3, oh_pct, severe_area,
            bool(thin_wall_regions), min_wall_mm, is_watertight,
        )

        return MeshAnalysisResult(
            volume_cm3=volume_cm3,
            surface_area_cm2=surface_area_cm2,
            bounding_box_mm=tuple(float(e) for e in extents),
            face_count=len(mesh.faces),
            vertex_count=len(mesh.vertices),
            is_watertight=is_watertight,
            is_manifold=is_watertight,
            overhang_regions=overhang_regions,
            total_overhang_area_mm2=total_oh_area,
            overhang_percentage=oh_pct,
            severe_overhang_mm2=severe_area,
            thin_wall_regions=thin_wall_regions,
            min_wall_thickness_mm=min_wall_mm,
            has_thin_walls=bool(thin_wall_regions),
            support_volume_cm3=support_vol,
            support_contact_area_cm2=support_contact,
            estimated_support_grams=support_grams,
            difficulty_score=difficulty,
            difficulty_factors=factors,
            print_time_minutes=print_time,
            filament_grams=filament,
        )

    def _analyze_overhangs(self, mesh):
        normals = mesh.face_normals
        areas = mesh.area_faces if hasattr(mesh, "area_faces") else np.ones(len(normals))
        centroids = mesh.triangles_center if hasattr(mesh, "triangles_center") else None
        z_min = float(mesh.bounds[0][2]) if mesh.bounds is not None else 0.0

        regions: list[OverhangRegion] = []
        total_oh_area = 0.0
        severe_area = 0.0

        for i, (normal, area) in enumerate(zip(normals, areas)):
            z_comp = float(normal[2])
            if z_comp >= -0.707:  # < 45 deg from horizontal = not an overhang
                continue

            angle_rad = math.acos(min(abs(z_comp), 1.0))
            angle_deg = math.degrees(angle_rad)

            if angle_deg > 25:
                severity = "severe"
                severe_area += float(area)
            elif angle_deg > 15:
                severity = "moderate"
            else:
                severity = "mild"

            centroid = tuple(float(c) for c in centroids[i]) if centroids is not None else (0.0, 0.0, 0.0)
            height_above = centroid[2] - z_min if centroid[2] > z_min else 0.0

            regions.append(OverhangRegion(
                centroid_mm=centroid,
                area_mm2=float(area),
                angle_deg=angle_deg,
                height_above_plate_mm=height_above,
                severity=severity,
            ))
            total_oh_area += float(area)

        total_surface = float(np.sum(areas))
        oh_pct = (total_oh_area / total_surface * 100.0) if total_surface > 0 else 0.0
        return regions, total_oh_area, oh_pct, severe_area

    def _detect_thin_walls(self, mesh) -> tuple[list[ThinWallRegion], float]:
        min_wall_mm = float(mesh.bounding_box.extents.min())
        regions: list[ThinWallRegion] = []
        min_safe = MIN_WALL_NOZZLE_WIDTHS * self.nozzle_diameter_mm

        if min_wall_mm < min_safe * 0.5:
            severity = "critical"
        elif min_wall_mm < min_safe:
            severity = "warning"
        elif min_wall_mm < min_safe * 1.5:
            severity = "info"
        else:
            severity = "ok"

        if severity != "ok":
            regions.append(ThinWallRegion(thickness_mm=min_wall_mm, face_indices=[], severity=severity))

        return regions, min_wall_mm

    def _estimate_support(self, overhang_regions: list[OverhangRegion]):
        total_contact_area = 0.0
        estimated_volume_mm3 = 0.0

        for region in overhang_regions:
            if region.severity in ("moderate", "severe"):
                cos_angle = math.cos(math.radians(region.angle_deg))
                contact = region.area_mm2 * cos_angle
                total_contact_area += contact
                sparsity = 0.35
                volume = region.height_above_plate_mm * contact * sparsity
                estimated_volume_mm3 += volume

        support_vol_cm3 = estimated_volume_mm3 / 1000.0
        contact_cm2 = total_contact_area / 100.0
        support_grams = support_vol_cm3 * SUPPORT_DENSITY
        return support_vol_cm3, contact_cm2, support_grams

    def _compute_difficulty(self, mesh, volume_cm3, oh_pct, severe_area, has_thin_walls, min_wall_mm, is_watertight):
        score = 1.0
        factors: list[str] = []

        if not is_watertight:
            score += 2.0
            factors.append("Mesh is not watertight")

        if oh_pct > 30:
            score += 2.0
            factors.append(f"Heavy overhangs ({oh_pct:.1f}%)")
        elif oh_pct > 15:
            score += 1.0
            factors.append(f"Moderate overhangs ({oh_pct:.1f}%)")
        elif oh_pct > 5:
            score += 0.5
            factors.append(f"Minor overhangs ({oh_pct:.1f}%)")

        if severe_area > 500:
            score += 1.0
            factors.append(f"Severe overhang area ({severe_area:.0f}mm²)")

        min_safe = MIN_WALL_NOZZLE_WIDTHS * self.nozzle_diameter_mm
        if min_wall_mm < min_safe * 0.5:
            score += 2.0
            factors.append(f"Critical thin walls ({min_wall_mm:.2f}mm)")
        elif min_wall_mm < min_safe:
            score += 1.0
            factors.append(f"Thin walls ({min_wall_mm:.2f}mm)")

        face_count = len(mesh.faces)
        if face_count > 500_000:
            score += 2.5
            factors.append(f"High complexity ({face_count:,} faces)")
        elif face_count > 200_000:
            score += 1.5
            factors.append(f"Medium-high complexity ({face_count:,} faces)")
        elif face_count > 100_000:
            score += 1.0
            factors.append(f"Moderate complexity ({face_count:,} faces)")

        extents = mesh.bounding_box.extents
        aspect = float(max(extents) / max(min(extents), 1e-6))
        if aspect > 5:
            score += 0.5
            factors.append(f"Extreme aspect ratio ({aspect:.1f})")

        difficulty = min(10.0, max(1.0, score))

        height_mm = float(extents[2])
        num_layers = int(math.ceil(height_mm / self.layer_height_mm))
        perimeter_mm = 2 * (float(extents[0]) + float(extents[1]))
        shell_time = perimeter_mm * 2 * num_layers / 3600
        vol_time = volume_cm3 * self.infill * 10 / 3600
        print_time_min = (shell_time + vol_time + height_mm / 600) * 60

        bbox_vol_cm3 = (float(extents[0]) * float(extents[1]) * float(extents[2])) / 1000.0
        filament_cm3 = bbox_vol_cm3 * (0.3 + self.infill * 0.4)
        filament_grams = filament_cm3 * self.material_density

        return difficulty, factors, print_time_min, filament_grams


def analyze_mesh(mesh, nozzle_diameter_mm: float = DEFAULT_NOZZLE_MM,
                 layer_height_mm: float = 0.2, infill: float = 0.15) -> dict:
    import trimesh
    if isinstance(mesh, (str, Path)):
        mesh = trimesh.load(str(mesh), force="mesh")
    analyzer = MeshAnalyzer(nozzle_diameter_mm=nozzle_diameter_mm,
                           layer_height_mm=layer_height_mm, infill=infill)
    result = analyzer.analyze(mesh)
    return result.summary()
