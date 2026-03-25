"""Multi-printer profiles database for PrintForge."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PrinterProfile:
    """Detailed printer specification."""
    name: str
    build_volume: Tuple[float, float, float]  # x, y, z in mm
    max_speed: float  # mm/s
    heated_bed: bool
    auto_level: bool
    nozzle_sizes: List[float]  # mm
    default_layer_height: float  # mm
    default_infill: float  # 0.0-1.0


PRINTER_DB: Dict[str, PrinterProfile] = {
    "bambu_x1c": PrinterProfile(
        name="Bambu Lab X1 Carbon",
        build_volume=(256, 256, 256),
        max_speed=500.0,
        heated_bed=True,
        auto_level=True,
        nozzle_sizes=[0.2, 0.4, 0.6, 0.8],
        default_layer_height=0.2,
        default_infill=0.15,
    ),
    "bambu_p1s": PrinterProfile(
        name="Bambu Lab P1S",
        build_volume=(256, 256, 256),
        max_speed=500.0,
        heated_bed=True,
        auto_level=True,
        nozzle_sizes=[0.2, 0.4, 0.6, 0.8],
        default_layer_height=0.2,
        default_infill=0.15,
    ),
    "bambu_a1": PrinterProfile(
        name="Bambu Lab A1",
        build_volume=(256, 256, 256),
        max_speed=500.0,
        heated_bed=True,
        auto_level=True,
        nozzle_sizes=[0.2, 0.4, 0.6, 0.8],
        default_layer_height=0.2,
        default_infill=0.15,
    ),
    "bambu_a1_mini": PrinterProfile(
        name="Bambu Lab A1 Mini",
        build_volume=(180, 180, 180),
        max_speed=500.0,
        heated_bed=True,
        auto_level=True,
        nozzle_sizes=[0.2, 0.4],
        default_layer_height=0.2,
        default_infill=0.15,
    ),
    "prusa_mk4": PrinterProfile(
        name="Prusa MK4",
        build_volume=(250, 210, 220),
        max_speed=200.0,
        heated_bed=True,
        auto_level=True,
        nozzle_sizes=[0.25, 0.4, 0.6, 0.8],
        default_layer_height=0.2,
        default_infill=0.15,
    ),
    "prusa_mini": PrinterProfile(
        name="Prusa Mini+",
        build_volume=(180, 180, 180),
        max_speed=200.0,
        heated_bed=True,
        auto_level=True,
        nozzle_sizes=[0.25, 0.4, 0.6],
        default_layer_height=0.2,
        default_infill=0.15,
    ),
    "creality_ender3": PrinterProfile(
        name="Creality Ender-3 V3",
        build_volume=(220, 220, 250),
        max_speed=250.0,
        heated_bed=True,
        auto_level=False,
        nozzle_sizes=[0.4, 0.6, 0.8],
        default_layer_height=0.2,
        default_infill=0.20,
    ),
    "creality_k1": PrinterProfile(
        name="Creality K1",
        build_volume=(220, 220, 250),
        max_speed=600.0,
        heated_bed=True,
        auto_level=True,
        nozzle_sizes=[0.4, 0.6, 0.8],
        default_layer_height=0.2,
        default_infill=0.15,
    ),
}


def get_profile(name: str) -> PrinterProfile:
    """Get a printer profile by key name.

    Args:
        name: Key from PRINTER_DB (e.g. 'bambu_x1c', 'prusa_mk4').

    Returns:
        PrinterProfile for the requested printer.

    Raises:
        KeyError: If the printer name is not found.
    """
    if name not in PRINTER_DB:
        available = ", ".join(sorted(PRINTER_DB.keys()))
        raise KeyError(f"Unknown printer '{name}'. Available: {available}")
    return PRINTER_DB[name]


def list_profiles() -> List[str]:
    """Return a sorted list of all available printer profile keys."""
    return sorted(PRINTER_DB.keys())
