"""Material database for PrintForge v2.2.

Properties for different 3D printing materials — used for cost estimation,
print settings, and compatibility checks.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Material:
    name: str
    type: str  # FDM, SLA, SLS
    density_g_cm3: float
    cost_per_kg_usd: float
    print_temp_c: int
    bed_temp_c: int
    max_overhang_deg: int  # max unsupported overhang angle
    min_wall_mm: float
    layer_height_range: tuple  # (min, max) in mm
    color_options: list
    properties: str  # brief description


MATERIALS: Dict[str, Material] = {
    "pla": Material(
        name="PLA", type="FDM", density_g_cm3=1.24, cost_per_kg_usd=20,
        print_temp_c=210, bed_temp_c=60, max_overhang_deg=45,
        min_wall_mm=0.4, layer_height_range=(0.1, 0.3),
        color_options=["white", "black", "red", "blue", "green", "yellow", "orange", "transparent"],
        properties="Easy to print, biodegradable, low warp. Best for prototyping and decorative parts.",
    ),
    "petg": Material(
        name="PETG", type="FDM", density_g_cm3=1.27, cost_per_kg_usd=25,
        print_temp_c=235, bed_temp_c=80, max_overhang_deg=40,
        min_wall_mm=0.4, layer_height_range=(0.1, 0.3),
        color_options=["transparent", "white", "black", "blue"],
        properties="Strong, flexible, chemical resistant. Good for functional parts.",
    ),
    "abs": Material(
        name="ABS", type="FDM", density_g_cm3=1.04, cost_per_kg_usd=22,
        print_temp_c=245, bed_temp_c=100, max_overhang_deg=40,
        min_wall_mm=0.5, layer_height_range=(0.1, 0.3),
        color_options=["white", "black", "red", "blue", "yellow"],
        properties="Tough, heat resistant, sandable. Needs enclosure. Good for engineering parts.",
    ),
    "tpu": Material(
        name="TPU", type="FDM", density_g_cm3=1.21, cost_per_kg_usd=35,
        print_temp_c=225, bed_temp_c=50, max_overhang_deg=35,
        min_wall_mm=0.6, layer_height_range=(0.15, 0.3),
        color_options=["white", "black", "transparent"],
        properties="Flexible, rubber-like. For phone cases, gaskets, wearables.",
    ),
    "nylon": Material(
        name="Nylon", type="FDM", density_g_cm3=1.14, cost_per_kg_usd=45,
        print_temp_c=260, bed_temp_c=80, max_overhang_deg=35,
        min_wall_mm=0.5, layer_height_range=(0.1, 0.25),
        color_options=["natural", "black"],
        properties="Very strong, low friction, chemical resistant. For gears, bearings, tools.",
    ),
    "resin_standard": Material(
        name="Standard Resin", type="SLA", density_g_cm3=1.12, cost_per_kg_usd=40,
        print_temp_c=0, bed_temp_c=0, max_overhang_deg=30,
        min_wall_mm=0.3, layer_height_range=(0.025, 0.1),
        color_options=["grey", "white", "black", "transparent", "green"],
        properties="High detail, smooth surface. For miniatures, jewelry masters, dental models.",
    ),
    "resin_tough": Material(
        name="Tough Resin", type="SLA", density_g_cm3=1.15, cost_per_kg_usd=60,
        print_temp_c=0, bed_temp_c=0, max_overhang_deg=30,
        min_wall_mm=0.4, layer_height_range=(0.025, 0.1),
        color_options=["grey", "black"],
        properties="ABS-like toughness with SLA detail. For functional prototypes.",
    ),
}


def get_material(name: str) -> Optional[Material]:
    return MATERIALS.get(name.lower())


def list_materials() -> list:
    return [
        {
            "id": k,
            "name": v.name,
            "type": v.type,
            "cost_per_kg": v.cost_per_kg_usd,
            "min_wall_mm": v.min_wall_mm,
            "properties": v.properties,
        }
        for k, v in MATERIALS.items()
    ]


def estimate_material_cost(volume_cm3: float, material_id: str = "pla") -> dict:
    """Estimate material cost for a given volume."""
    mat = get_material(material_id)
    if not mat:
        return {"error": f"Unknown material: {material_id}"}
    
    weight_g = volume_cm3 * mat.density_g_cm3
    cost_usd = (weight_g / 1000) * mat.cost_per_kg_usd

    return {
        "material": mat.name,
        "volume_cm3": round(volume_cm3, 2),
        "weight_g": round(weight_g, 1),
        "cost_usd": round(cost_usd, 2),
        "density": mat.density_g_cm3,
        "cost_per_kg": mat.cost_per_kg_usd,
    }
