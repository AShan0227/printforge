"""Mesh Quality Scoring: Analyze a generated mesh and return a quality score 0-100."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Detailed quality breakdown for a mesh."""
    total_score: float  # 0-100

    # Individual metrics
    watertight_score: float  # 0-30
    face_count_score: float  # 0-20
    aspect_ratio_score: float  # 0-15
    thin_wall_score: float  # 0-20
    overhang_score: float  # 0-15

    is_watertight: bool
    face_count: int
    aspect_ratio: float
    min_thickness_mm: float
    overhang_percentage: float

    @property
    def grade(self) -> str:
        if self.total_score >= 90:
            return "A"
        elif self.total_score >= 75:
            return "B"
        elif self.total_score >= 60:
            return "C"
        elif self.total_score >= 40:
            return "D"
        else:
            return "F"


class QualityScorer:
    """Analyze a trimesh mesh and produce a quality score from 0 to 100.

    Scoring breakdown:
        - Watertight:           30 pts
        - Face count ratio:     20 pts
        - Aspect ratio:         15 pts
        - Thin wall detection:  20 pts
        - Overhang percentage:  15 pts
    """

    # Ideal face count range for a good-quality printable model
    IDEAL_FACE_MIN = 5_000
    IDEAL_FACE_MAX = 200_000

    # Max acceptable aspect ratio (longest / shortest extent)
    MAX_GOOD_ASPECT = 10.0

    # Minimum wall thickness (mm) considered safe for FDM
    MIN_WALL_MM = 0.8

    # Overhang threshold (degrees from vertical)
    OVERHANG_ANGLE_DEG = 45.0

    def score(self, mesh) -> QualityReport:
        """Score the given trimesh mesh.

        Args:
            mesh: A trimesh.Trimesh object.

        Returns:
            QualityReport with total_score and per-metric breakdown.
        """
        import trimesh

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh).__name__}")

        # --- Watertight (30 pts) ---
        is_watertight = bool(mesh.is_watertight)
        watertight_score = 30.0 if is_watertight else 0.0

        # --- Face count ratio (20 pts) ---
        face_count = len(mesh.faces)
        face_count_score = self._score_face_count(face_count)

        # --- Aspect ratio (15 pts) ---
        extents = mesh.bounding_box.extents
        aspect_ratio = float(extents.max() / max(extents.min(), 1e-6))
        aspect_ratio_score = self._score_aspect_ratio(aspect_ratio)

        # --- Thin wall detection (20 pts) ---
        min_thickness_mm = float(extents.min())
        thin_wall_score = self._score_thin_walls(mesh)

        # --- Overhang percentage (15 pts) ---
        overhang_pct = self._compute_overhang_percentage(mesh)
        overhang_score = self._score_overhang(overhang_pct)

        total = (
            watertight_score
            + face_count_score
            + aspect_ratio_score
            + thin_wall_score
            + overhang_score
        )

        return QualityReport(
            total_score=round(total, 1),
            watertight_score=round(watertight_score, 1),
            face_count_score=round(face_count_score, 1),
            aspect_ratio_score=round(aspect_ratio_score, 1),
            thin_wall_score=round(thin_wall_score, 1),
            overhang_score=round(overhang_score, 1),
            is_watertight=is_watertight,
            face_count=face_count,
            aspect_ratio=round(aspect_ratio, 2),
            min_thickness_mm=round(min_thickness_mm, 2),
            overhang_percentage=round(overhang_pct, 1),
        )

    def _score_face_count(self, count: int) -> float:
        """20 pts: ideal range [IDEAL_FACE_MIN, IDEAL_FACE_MAX]. Degrade outside."""
        if self.IDEAL_FACE_MIN <= count <= self.IDEAL_FACE_MAX:
            return 20.0
        if count < self.IDEAL_FACE_MIN:
            ratio = count / self.IDEAL_FACE_MIN
            return max(0.0, 20.0 * ratio)
        # count > IDEAL_FACE_MAX
        over_ratio = self.IDEAL_FACE_MAX / count
        return max(0.0, 20.0 * over_ratio)

    def _score_aspect_ratio(self, ratio: float) -> float:
        """15 pts: 1.0 is perfect, degrade linearly to MAX_GOOD_ASPECT."""
        if ratio <= 1.0:
            return 15.0
        if ratio >= self.MAX_GOOD_ASPECT:
            return 0.0
        return 15.0 * (1.0 - (ratio - 1.0) / (self.MAX_GOOD_ASPECT - 1.0))

    def _score_thin_walls(self, mesh) -> float:
        """20 pts: check minimum extent of individual connected components.

        For simplicity, uses the bounding box min extent of the whole mesh.
        """
        min_extent = float(mesh.bounding_box.extents.min())
        if min_extent >= self.MIN_WALL_MM:
            return 20.0
        if min_extent <= 0:
            return 0.0
        return 20.0 * (min_extent / self.MIN_WALL_MM)

    def _compute_overhang_percentage(self, mesh) -> float:
        """Compute the percentage of faces that overhang beyond the threshold angle."""
        normals = mesh.face_normals
        # Overhang: face normal points downward beyond threshold
        # Dot with up vector (0, 0, 1): negative means pointing down
        z_component = normals[:, 2]
        threshold = -np.cos(np.radians(self.OVERHANG_ANGLE_DEG))
        overhang_mask = z_component < threshold
        if len(normals) == 0:
            return 0.0
        return float(overhang_mask.sum() / len(normals) * 100.0)

    def _score_overhang(self, pct: float) -> float:
        """15 pts: 0% overhang → 15, 50%+ → 0."""
        if pct <= 0:
            return 15.0
        if pct >= 50.0:
            return 0.0
        return 15.0 * (1.0 - pct / 50.0)
