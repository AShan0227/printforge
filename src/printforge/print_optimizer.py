"""Print Optimizer: Orientation, time estimation, material estimation, printability checks."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Common printer presets (build volume in mm)
PRINTER_PRESETS = {
    "bambu-a1": {"volume": (256, 256, 256), "name": "Bambu Lab A1"},
    "bambu-a1-mini": {"volume": (180, 180, 180), "name": "Bambu Lab A1 Mini"},
    "bambu-p1s": {"volume": (256, 256, 256), "name": "Bambu Lab P1S"},
    "bambu-x1c": {"volume": (256, 256, 256), "name": "Bambu Lab X1 Carbon"},
    "prusa-mk4": {"volume": (250, 210, 220), "name": "Prusa MK4"},
    "ender-3": {"volume": (220, 220, 250), "name": "Creality Ender 3"},
    "voron-0.2": {"volume": (120, 120, 120), "name": "Voron 0.2"},
}

# Material densities in g/cm^3
MATERIAL_DENSITY = {
    "pla": 1.24,
    "petg": 1.27,
    "abs": 1.04,
    "tpu": 1.21,
    "asa": 1.07,
    "nylon": 1.14,
}


@dataclass
class Orientation:
    """A candidate print orientation."""
    rotation_matrix: np.ndarray
    support_volume_estimate: float  # mm^3 of estimated support
    base_area: float               # mm^2 of contact with build plate
    height: float                  # mm print height
    score: float                   # lower is better
    overhang_percentage: float = 0.0  # % of total face area that overhangs
    support_contact_area: float = 0.0  # mm^2 of support contact with model


@dataclass
class PrintabilityIssue:
    """A printability issue found during analysis."""
    severity: str  # "error", "warning", "info"
    category: str  # "volume", "overhang", "thin_wall", "watertight"
    message: str


@dataclass
class PrintEstimate:
    """Estimated print time and material usage."""
    print_time_minutes: float
    filament_grams: float
    filament_meters: float
    layer_count: int


class PrintOptimizer:
    """Analyze and optimize meshes for 3D printing."""

    def find_best_orientation(self, mesh, num_candidates: int = 24) -> Orientation:
        """Find the rotation that minimizes support material.

        Analyzes mesh face normals for 6 cardinal rotations (±X, ±Y, ±Z as up)
        plus additional candidates. Scores based on overhanging face area and
        support contact area.

        Overhang: face normal Z component < -0.5 (>45° from vertical).
        Score: total_overhang_area × 0.7 + total_support_contact_area × 0.3
        """
        import trimesh

        best = None

        # Always include the 6 cardinal rotations (±X, ±Y, ±Z as up)
        cardinal_rotations = self._generate_cardinal_rotations()
        extra_rotations = self._generate_rotations(num_candidates)

        # Deduplicate: use cardinals first, then fill with extras
        all_rotations = list(cardinal_rotations)
        for rot in extra_rotations:
            if len(all_rotations) >= max(num_candidates, 6):
                break
            # Skip if nearly identical to an existing cardinal
            is_dup = False
            for existing in cardinal_rotations:
                if np.allclose(rot, existing, atol=1e-6):
                    is_dup = True
                    break
            if not is_dup:
                all_rotations.append(rot)

        for rotation in all_rotations:
            rotated = mesh.copy()
            rotated.apply_transform(rotation)

            score_info = self._score_orientation(rotated, rotation)

            if best is None or score_info.score < best.score:
                best = score_info

        logger.info(
            f"Best orientation: height={best.height:.1f}mm, "
            f"support_est={best.support_volume_estimate:.0f}mm^3, "
            f"overhang={best.overhang_percentage:.1f}%, score={best.score:.2f}"
        )
        return best

    def _generate_cardinal_rotations(self) -> list[np.ndarray]:
        """Generate the 6 cardinal rotations (±X, ±Y, ±Z as up direction)."""
        import trimesh

        rotations = []
        # +Z up (identity — default orientation)
        rotations.append(np.eye(4))
        # -Z up (flip upside down — 180° around X)
        rotations.append(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
        # +X up (90° around Y)
        rotations.append(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        # -X up (-90° around Y)
        rotations.append(trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 1, 0]))
        # +Y up (-90° around X)
        rotations.append(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        # -Y up (90° around X)
        rotations.append(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
        return rotations

    def estimate_print_time(
        self,
        mesh,
        layer_height: float = 0.2,
        print_speed: float = 60.0,
        travel_speed: float = 150.0,
    ) -> float:
        """Estimate print time in minutes.

        Uses a simplified model based on:
        - Number of layers
        - Perimeter length per layer (approximated)
        - Infill travel distance
        """
        extents = mesh.bounding_box.extents
        height = extents[2]  # print height in mm
        num_layers = int(np.ceil(height / layer_height))

        # Approximate cross-section area and perimeter
        area_xy = extents[0] * extents[1]
        perimeter = 2 * (extents[0] + extents[1])

        # Estimate time per layer
        # Perimeter time: 2 walls * perimeter / speed
        perimeter_time = (2 * perimeter) / print_speed  # minutes per layer

        # Infill time: approximate fill path length
        infill_time = (area_xy * 0.15) / (print_speed * layer_height * 0.4) * 0.001

        # Travel time between features
        travel_time = perimeter / travel_speed * 0.5

        total_minutes = num_layers * (perimeter_time + infill_time + travel_time)

        # Add overhead for heating, first layer, z-hops, retractions
        total_minutes *= 1.15
        total_minutes += 3.0  # heating/calibration

        return round(total_minutes, 1)

    def estimate_material(
        self,
        mesh,
        infill: float = 0.15,
        wall_count: int = 2,
        layer_height: float = 0.2,
        material: str = "pla",
        filament_diameter: float = 1.75,
    ) -> PrintEstimate:
        """Estimate filament usage in grams and meters.

        Args:
            mesh: The mesh to estimate for.
            infill: Infill density (0.0-1.0).
            wall_count: Number of perimeter walls.
            layer_height: Layer height in mm.
            material: Material type for density lookup.
            filament_diameter: Filament diameter in mm.
        """
        extents = mesh.bounding_box.extents
        height = extents[2]
        num_layers = int(np.ceil(height / layer_height))

        # Volume calculation
        # Outer shell volume
        nozzle_width = 0.4  # mm
        shell_thickness = wall_count * nozzle_width

        # Approximate shell volume (surface area * shell thickness)
        surface_area = mesh.area if hasattr(mesh, "area") else 2 * (
            extents[0] * extents[1] + extents[1] * extents[2] + extents[0] * extents[2]
        )
        shell_volume_mm3 = surface_area * shell_thickness

        # Infill volume (interior volume * infill density)
        bbox_volume = extents[0] * extents[1] * extents[2]
        interior_volume = max(0, bbox_volume - shell_volume_mm3)
        infill_volume_mm3 = interior_volume * infill

        # Top/bottom solid layers (typically 3-4 layers each)
        solid_layers = 4
        solid_volume = extents[0] * extents[1] * layer_height * solid_layers * 2

        total_volume_mm3 = shell_volume_mm3 + infill_volume_mm3 + solid_volume
        total_volume_cm3 = total_volume_mm3 / 1000.0

        # Weight
        density = MATERIAL_DENSITY.get(material.lower(), 1.24)
        weight_grams = total_volume_cm3 * density

        # Filament length
        filament_area = np.pi * (filament_diameter / 2) ** 2  # mm^2
        filament_length_mm = total_volume_mm3 / filament_area
        filament_length_m = filament_length_mm / 1000.0

        return PrintEstimate(
            print_time_minutes=self.estimate_print_time(mesh, layer_height),
            filament_grams=round(weight_grams, 1),
            filament_meters=round(filament_length_m, 2),
            layer_count=num_layers,
        )

    def check_printability(
        self,
        mesh,
        build_volume: tuple[float, float, float] = (256, 256, 256),
    ) -> list[PrintabilityIssue]:
        """Check a mesh for common 3D printing issues.

        Returns a list of issues found.
        """
        issues = []
        extents = mesh.bounding_box.extents

        # 1. Check build volume
        if (extents[0] > build_volume[0] or
                extents[1] > build_volume[1] or
                extents[2] > build_volume[2]):
            issues.append(PrintabilityIssue(
                severity="error",
                category="volume",
                message=(
                    f"Model ({extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f}mm) "
                    f"exceeds build volume ({build_volume[0]} x {build_volume[1]} x {build_volume[2]}mm)"
                ),
            ))

        # 2. Check watertight
        if hasattr(mesh, "is_watertight") and not mesh.is_watertight:
            issues.append(PrintabilityIssue(
                severity="error",
                category="watertight",
                message="Mesh is not watertight. Slicers may produce artifacts.",
            ))

        # 3. Check for very thin features
        min_extent = extents.min()
        if min_extent < 0.4:
            issues.append(PrintabilityIssue(
                severity="warning",
                category="thin_wall",
                message=f"Minimum dimension is {min_extent:.2f}mm, below 0.4mm nozzle width.",
            ))

        # 4. Check for degenerate faces
        if hasattr(mesh, "face_adjacency") and len(mesh.faces) > 0:
            area = mesh.area
            if area < 1.0:
                issues.append(PrintabilityIssue(
                    severity="warning",
                    category="thin_wall",
                    message=f"Very small surface area ({area:.2f}mm^2). Model may be too small to print.",
                ))

        # 5. Check face count
        if len(mesh.faces) > 500_000:
            issues.append(PrintabilityIssue(
                severity="info",
                category="complexity",
                message=f"High face count ({len(mesh.faces):,}). Consider simplifying for faster slicing.",
            ))

        # 6. Overhang estimate (simplified)
        face_normals = mesh.face_normals if hasattr(mesh, "face_normals") else None
        if face_normals is not None and len(face_normals) > 0:
            # Faces pointing significantly downward need support
            down_component = face_normals[:, 2]
            overhang_ratio = np.sum(down_component < -0.7) / len(face_normals)
            if overhang_ratio > 0.10:
                issues.append(PrintabilityIssue(
                    severity="warning",
                    category="overhang",
                    message=f"{overhang_ratio*100:.0f}% of faces are steep overhangs (>45deg). Consider rotating or adding supports.",
                ))

        return issues

    def _generate_rotations(self, n: int) -> list[np.ndarray]:
        """Generate candidate rotation matrices."""
        import trimesh

        rotations = [np.eye(4)]  # identity

        # 90-degree rotations around each axis
        for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            for angle in (np.pi / 2, np.pi, 3 * np.pi / 2):
                mat = trimesh.transformations.rotation_matrix(angle, axis)
                rotations.append(mat)

        # Diagonal rotations
        for angle in (np.pi / 4, 3 * np.pi / 4):
            for axis in ([1, 1, 0], [1, 0, 1], [0, 1, 1]):
                norm_axis = np.array(axis, dtype=float)
                norm_axis /= np.linalg.norm(norm_axis)
                mat = trimesh.transformations.rotation_matrix(angle, norm_axis)
                rotations.append(mat)

        return rotations[:n]

    def _score_orientation(self, mesh, rotation_matrix: np.ndarray) -> Orientation:
        """Score a mesh orientation for printability using face normal analysis.

        Overhang = face normal Z component < -0.5 (more than 45° from vertical).
        Score = total_overhang_area × 0.7 + support_contact_area × 0.3
        """
        extents = mesh.bounding_box.extents
        height = extents[2]
        base_area = extents[0] * extents[1]

        overhang_area = 0.0
        support_contact_area = 0.0
        total_face_area = 0.0
        support_volume = 0.0

        if hasattr(mesh, "face_normals") and len(mesh.face_normals) > 0:
            normals = mesh.face_normals
            areas = mesh.area_faces if hasattr(mesh, "area_faces") else np.ones(len(normals))
            total_face_area = float(np.sum(areas))

            # Overhanging faces: Z component of normal < -0.5
            down_mask = normals[:, 2] < -0.5
            if np.any(down_mask):
                overhang_area = float(np.sum(areas[down_mask]))

                # Support contact area = projected overhang area onto XY plane
                # Weight by how much each face points downward
                down_components = np.abs(normals[down_mask, 2])
                support_contact_area = float(np.sum(areas[down_mask] * down_components))

                # Estimate support volume from overhanging faces
                centroids = mesh.triangles_center[down_mask] if hasattr(mesh, "triangles_center") else None
                if centroids is not None:
                    heights_above_plate = centroids[:, 2] - mesh.bounds[0][2]
                    support_volume = float(np.sum(areas[down_mask] * heights_above_plate))
                else:
                    support_volume = float(overhang_area * height * 0.3)

        overhang_percentage = (overhang_area / total_face_area * 100.0) if total_face_area > 0 else 0.0

        # Score: weighted combination of overhang area and support contact area
        score = overhang_area * 0.7 + support_contact_area * 0.3

        return Orientation(
            rotation_matrix=rotation_matrix,
            support_volume_estimate=support_volume,
            base_area=base_area,
            height=height,
            score=score,
            overhang_percentage=overhang_percentage,
            support_contact_area=support_contact_area,
        )
