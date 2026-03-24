"""Cost Estimator: Calculate printing costs including filament, time, and electricity."""

import logging
from dataclasses import dataclass

import numpy as np

from .print_optimizer import MATERIAL_DENSITY

logger = logging.getLogger(__name__)

# Material cost per kg in USD
MATERIAL_COST_PER_KG = {
    "pla": 20.0,
    "petg": 25.0,
    "abs": 22.0,
    "tpu": 35.0,
    "asa": 28.0,
    "nylon": 40.0,
}

# Electricity rate in USD per kWh
DEFAULT_ELECTRICITY_RATE = 0.12

# Printer power consumption in watts
DEFAULT_PRINTER_WATTAGE = 150


@dataclass
class CostEstimate:
    """Full cost breakdown for a 3D print."""
    filament_grams: float
    filament_meters: float
    filament_cost_usd: float
    print_time_hours: float
    electricity_cost_usd: float
    total_cost_usd: float


class CostEstimator:
    """Estimate the total cost of printing a mesh."""

    def __init__(
        self,
        electricity_rate: float = DEFAULT_ELECTRICITY_RATE,
        printer_wattage: float = DEFAULT_PRINTER_WATTAGE,
    ):
        self.electricity_rate = electricity_rate
        self.printer_wattage = printer_wattage

    def estimate(
        self,
        mesh,
        material: str = "PLA",
        infill: float = 0.15,
        layer_height: float = 0.2,
        filament_diameter: float = 1.75,
    ) -> CostEstimate:
        """Estimate the full cost of printing a mesh.

        Args:
            mesh: Trimesh mesh object.
            material: Material type (PLA, PETG, ABS, TPU, etc.).
            infill: Infill density (0.0-1.0).
            layer_height: Layer height in mm.
            filament_diameter: Filament diameter in mm.

        Returns:
            CostEstimate with full cost breakdown.
        """
        mat_key = material.lower()
        density = MATERIAL_DENSITY.get(mat_key, 1.24)  # g/cm³
        cost_per_kg = MATERIAL_COST_PER_KG.get(mat_key, 20.0)

        # --- Filament usage ---
        # Use mesh volume (in mm³) if available, otherwise approximate from bbox
        if hasattr(mesh, "volume") and mesh.volume > 0:
            mesh_volume_mm3 = abs(mesh.volume)
        else:
            extents = mesh.bounding_box.extents
            mesh_volume_mm3 = extents[0] * extents[1] * extents[2]

        # Effective volume = mesh_volume × infill
        # (shell is solid, but we simplify by using infill on entire volume)
        effective_volume_mm3 = mesh_volume_mm3 * infill
        effective_volume_cm3 = effective_volume_mm3 / 1000.0

        filament_grams = effective_volume_cm3 * density

        # Filament length from volume
        filament_cross_area_mm2 = np.pi * (filament_diameter / 2) ** 2
        filament_meters = (effective_volume_mm3 / filament_cross_area_mm2) / 1000.0

        filament_cost_usd = (filament_grams / 1000.0) * cost_per_kg

        # --- Print time ---
        extents = mesh.bounding_box.extents
        height_mm = extents[2]
        num_layers = int(np.ceil(height_mm / layer_height))

        # Per-layer time estimate
        perimeter = 2 * (extents[0] + extents[1])
        area_xy = extents[0] * extents[1]
        print_speed = 60.0  # mm/s

        # Perimeter time: 2 walls
        perimeter_time_s = (2 * perimeter) / print_speed
        # Infill time: fill area at given density
        nozzle_width = 0.4  # mm
        infill_line_spacing = nozzle_width / infill if infill > 0 else 1e6
        infill_path_length = area_xy / infill_line_spacing if infill > 0 else 0
        infill_time_s = infill_path_length / print_speed

        layer_time_s = perimeter_time_s + infill_time_s
        total_time_s = num_layers * layer_time_s
        # Overhead: heating, first layer, retractions, z-hops
        total_time_s *= 1.15
        total_time_s += 180  # 3 min warmup

        print_time_hours = total_time_s / 3600.0

        # --- Electricity cost ---
        electricity_kwh = (self.printer_wattage / 1000.0) * print_time_hours
        electricity_cost_usd = electricity_kwh * self.electricity_rate

        total_cost_usd = filament_cost_usd + electricity_cost_usd

        return CostEstimate(
            filament_grams=round(filament_grams, 1),
            filament_meters=round(filament_meters, 2),
            filament_cost_usd=round(filament_cost_usd, 2),
            print_time_hours=round(print_time_hours, 2),
            electricity_cost_usd=round(electricity_cost_usd, 2),
            total_cost_usd=round(total_cost_usd, 2),
        )
