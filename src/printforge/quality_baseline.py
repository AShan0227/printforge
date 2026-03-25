"""Quality baseline data: expected scores and guidance for different object types."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
