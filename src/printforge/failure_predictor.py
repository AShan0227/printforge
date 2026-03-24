"""Print Failure Prediction: Analyze mesh geometry to predict print failures."""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FailureRisk:
    """A single identified failure risk."""
    type: str          # e.g. "thin_wall", "steep_overhang", "long_bridge", etc.
    severity: str      # "low", "medium", "high", "critical"
    location: str      # human-readable description of where
    suggestion: str    # recommended fix


@dataclass
class FailurePrediction:
    """Result of failure prediction analysis."""
    risk_score: float  # 0-100 (0 = safe, 100 = will definitely fail)
    risks: List[FailureRisk] = field(default_factory=list)

    @property
    def will_likely_fail(self) -> bool:
        return self.risk_score >= 70

    @property
    def risk_level(self) -> str:
        if self.risk_score < 20:
            return "low"
        elif self.risk_score < 50:
            return "medium"
        elif self.risk_score < 70:
            return "high"
        else:
            return "critical"


# Printer presets: build volume (mm) and common capabilities
PRINTER_CONFIGS = {
    "a1": {"name": "Bambu A1", "volume": (256, 256, 256), "min_layer": 0.08},
    "a1-mini": {"name": "Bambu A1 Mini", "volume": (180, 180, 180), "min_layer": 0.08},
    "x1c": {"name": "Bambu X1C", "volume": (256, 256, 256), "min_layer": 0.08},
    "p1s": {"name": "Bambu P1S", "volume": (256, 256, 256), "min_layer": 0.08},
    "prusa-mk4": {"name": "Prusa MK4", "volume": (250, 210, 220), "min_layer": 0.05},
    "ender3": {"name": "Ender 3", "volume": (220, 220, 250), "min_layer": 0.1},
}


class FailurePredictor:
    """Analyze a mesh for potential 3D printing failure modes.

    Checks for:
    - Thin walls (< 0.4mm)
    - Steep overhangs (> 70°)
    - Long bridging distances (> 10mm)
    - High aspect ratio / topple risk (> 5:1)
    - Small features (< 0.2mm)
    - Disconnected parts (islands)
    """

    # Thresholds
    THIN_WALL_MM = 0.4
    OVERHANG_ANGLE_DEG = 70.0
    MAX_BRIDGE_MM = 10.0
    MAX_ASPECT_RATIO = 5.0
    SMALL_FEATURE_MM = 0.2

    def predict(
        self,
        mesh,
        material: str = "PLA",
        layer_height: float = 0.2,
        printer: str = "a1",
    ) -> FailurePrediction:
        """Predict print failures for a given mesh.

        Args:
            mesh: A trimesh.Trimesh object.
            material: Filament material (PLA, PETG, ABS, TPU).
            layer_height: Layer height in mm.
            printer: Printer preset key.

        Returns:
            FailurePrediction with risk_score and list of risks.
        """
        import trimesh

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh).__name__}")

        risks: List[FailureRisk] = []

        # 1. Thin walls
        risks.extend(self._check_thin_walls(mesh))

        # 2. Steep overhangs
        risks.extend(self._check_overhangs(mesh))

        # 3. Bridging distance
        risks.extend(self._check_bridging(mesh))

        # 4. Aspect ratio / topple risk
        risks.extend(self._check_aspect_ratio(mesh))

        # 5. Small features
        risks.extend(self._check_small_features(mesh))

        # 6. Island detection (disconnected parts)
        risks.extend(self._check_islands(mesh))

        # Calculate overall risk score
        risk_score = self._calculate_risk_score(risks)

        logger.info(
            "Failure prediction: score=%d, risks=%d, material=%s, printer=%s",
            risk_score, len(risks), material, printer,
        )

        return FailurePrediction(risk_score=risk_score, risks=risks)

    def _check_thin_walls(self, mesh) -> List[FailureRisk]:
        """Detect thin wall regions by checking minimum bounding box extents."""
        risks = []
        extents = mesh.bounding_box.extents
        min_extent = float(extents.min())

        if min_extent < self.THIN_WALL_MM:
            severity = "critical" if min_extent < 0.2 else "high"
            risks.append(FailureRisk(
                type="thin_wall",
                severity=severity,
                location=f"Minimum thickness: {min_extent:.2f}mm",
                suggestion=f"Increase wall thickness to at least {self.THIN_WALL_MM}mm. "
                           f"Consider scaling up or thickening thin regions.",
            ))
        elif min_extent < self.THIN_WALL_MM * 2:
            risks.append(FailureRisk(
                type="thin_wall",
                severity="medium",
                location=f"Minimum thickness: {min_extent:.2f}mm",
                suggestion="Wall thickness is borderline. Consider increasing for reliability.",
            ))
        return risks

    def _check_overhangs(self, mesh) -> List[FailureRisk]:
        """Detect steep overhang faces (> threshold angle from vertical)."""
        risks = []
        normals = mesh.face_normals
        if len(normals) == 0:
            return risks

        # Overhang angle: angle between face normal and downward vector
        # A face pointing straight down has dot product -1 with up vector
        z_component = normals[:, 2]
        threshold = -np.cos(np.radians(self.OVERHANG_ANGLE_DEG))
        overhang_mask = z_component < threshold

        overhang_pct = float(overhang_mask.sum() / len(normals) * 100.0)

        if overhang_pct > 30:
            risks.append(FailureRisk(
                type="steep_overhang",
                severity="critical",
                location=f"{overhang_pct:.1f}% of faces exceed {self.OVERHANG_ANGLE_DEG}° overhang",
                suggestion="Add supports or reorient the model. Consider splitting into parts.",
            ))
        elif overhang_pct > 10:
            risks.append(FailureRisk(
                type="steep_overhang",
                severity="high",
                location=f"{overhang_pct:.1f}% of faces exceed {self.OVERHANG_ANGLE_DEG}° overhang",
                suggestion="Enable tree supports in slicer. Consider reorienting the model.",
            ))
        elif overhang_pct > 3:
            risks.append(FailureRisk(
                type="steep_overhang",
                severity="medium",
                location=f"{overhang_pct:.1f}% of faces exceed {self.OVERHANG_ANGLE_DEG}° overhang",
                suggestion="Minor overhangs detected. Supports recommended for best quality.",
            ))
        return risks

    def _check_bridging(self, mesh) -> List[FailureRisk]:
        """Estimate bridging distance from downward-facing face spans."""
        risks = []
        normals = mesh.face_normals
        if len(normals) == 0:
            return risks

        # Find faces that are nearly horizontal and facing down
        z_component = normals[:, 2]
        bridge_mask = z_component < -0.95  # nearly flat, facing down

        if bridge_mask.sum() == 0:
            return risks

        # Estimate bridging distance from bounding box of bridge faces
        bridge_face_indices = np.where(bridge_mask)[0]
        bridge_vertices = mesh.vertices[mesh.faces[bridge_face_indices].flatten()]

        if len(bridge_vertices) == 0:
            return risks

        xy_span = bridge_vertices[:, :2]
        x_range = float(xy_span[:, 0].max() - xy_span[:, 0].min())
        y_range = float(xy_span[:, 1].max() - xy_span[:, 1].min())
        bridge_dist = max(x_range, y_range)

        if bridge_dist > self.MAX_BRIDGE_MM * 2:
            risks.append(FailureRisk(
                type="long_bridge",
                severity="critical",
                location=f"Estimated bridge span: {bridge_dist:.1f}mm",
                suggestion=f"Bridge exceeds {self.MAX_BRIDGE_MM}mm. Add supports under flat overhangs.",
            ))
        elif bridge_dist > self.MAX_BRIDGE_MM:
            risks.append(FailureRisk(
                type="long_bridge",
                severity="high",
                location=f"Estimated bridge span: {bridge_dist:.1f}mm",
                suggestion="Long bridge detected. Enable supports or redesign with chamfers.",
            ))
        return risks

    def _check_aspect_ratio(self, mesh) -> List[FailureRisk]:
        """Check for high aspect ratio (topple risk)."""
        risks = []
        extents = mesh.bounding_box.extents
        sorted_extents = sorted(extents, reverse=True)

        # Height (Z) vs base footprint
        height = float(extents[2])
        base_min = float(min(extents[0], extents[1]))

        if base_min < 1e-6:
            return risks

        aspect = height / base_min

        if aspect > self.MAX_ASPECT_RATIO * 2:
            risks.append(FailureRisk(
                type="topple_risk",
                severity="critical",
                location=f"Height-to-base ratio: {aspect:.1f}:1",
                suggestion="Very high topple risk. Add a brim/raft or widen the base.",
            ))
        elif aspect > self.MAX_ASPECT_RATIO:
            risks.append(FailureRisk(
                type="topple_risk",
                severity="high",
                location=f"Height-to-base ratio: {aspect:.1f}:1",
                suggestion="Tall and narrow model. Use a brim for better adhesion.",
            ))
        return risks

    def _check_small_features(self, mesh) -> List[FailureRisk]:
        """Detect very small geometric features."""
        risks = []
        extents = mesh.bounding_box.extents
        min_extent = float(extents.min())

        if min_extent < self.SMALL_FEATURE_MM:
            risks.append(FailureRisk(
                type="small_feature",
                severity="high",
                location=f"Feature size: {min_extent:.3f}mm (below {self.SMALL_FEATURE_MM}mm)",
                suggestion="Features below 0.2mm won't resolve on FDM printers. "
                           "Consider SLA/resin printing or scaling up.",
            ))
        return risks

    def _check_islands(self, mesh) -> List[FailureRisk]:
        """Detect disconnected parts (islands) in the mesh."""
        risks = []

        try:
            bodies = mesh.split(only_watertight=False)
            num_bodies = len(bodies)
        except Exception:
            num_bodies = 1

        if num_bodies > 5:
            risks.append(FailureRisk(
                type="islands",
                severity="critical",
                location=f"{num_bodies} disconnected parts detected",
                suggestion="Multiple disconnected parts will fail to print as one piece. "
                           "Join parts or print separately.",
            ))
        elif num_bodies > 1:
            risks.append(FailureRisk(
                type="islands",
                severity="high",
                location=f"{num_bodies} disconnected parts detected",
                suggestion="Disconnected parts may print separately and fall. "
                           "Consider joining or using supports.",
            ))
        return risks

    def _calculate_risk_score(self, risks: List[FailureRisk]) -> float:
        """Calculate overall risk score (0-100) from individual risks."""
        if not risks:
            return 0.0

        severity_weights = {
            "critical": 35,
            "high": 20,
            "medium": 10,
            "low": 5,
        }

        total = sum(severity_weights.get(r.severity, 5) for r in risks)
        return min(100.0, float(total))
