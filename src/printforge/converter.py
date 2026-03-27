"""Mesh format converter for PrintForge v2.2.

Supports: STL ↔ OBJ ↔ 3MF ↔ GLB ↔ PLY
"""

import logging
import os
from pathlib import Path
from typing import Optional

import trimesh

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"stl", "obj", "3mf", "glb", "gltf", "ply", "off"}


def convert_mesh(
    input_path: str,
    output_path: str,
    target_faces: Optional[int] = None,
    color: str = "#4CAF50",
) -> dict:
    """Convert a mesh file to another format.

    Args:
        input_path: source mesh file
        output_path: destination path (format inferred from extension)
        target_faces: if set, simplify mesh to this many faces
        color: hex color for GLB export

    Returns:
        dict with conversion stats
    """
    input_ext = Path(input_path).suffix.lstrip(".").lower()
    output_ext = Path(output_path).suffix.lstrip(".").lower()

    if input_ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported input format: {input_ext}")
    if output_ext not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported output format: {output_ext}")

    # Load
    mesh = trimesh.load(input_path, force="mesh")
    original_faces = len(mesh.faces)
    original_verts = len(mesh.vertices)

    # Simplify if requested
    if target_faces and len(mesh.faces) > target_faces:
        try:
            import fast_simplification
            ratio = 1.0 - (target_faces / len(mesh.faces))
            verts, faces = fast_simplification.simplify(
                mesh.vertices, mesh.faces, target_reduction=max(0.01, min(0.99, ratio))
            )
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        except ImportError:
            mesh = mesh.simplify_quadric_decimation(target_faces)
        logger.info(f"Simplified: {original_faces} → {len(mesh.faces)} faces")

    # Export
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if output_ext == "glb":
        from .export_glb import export_glb
        export_glb(mesh, output_path, color=color)
    else:
        mesh.export(output_path, file_type=output_ext)

    output_size = os.path.getsize(output_path)

    logger.info(f"Converted: {input_ext} → {output_ext} ({output_size} bytes)")

    return {
        "input_format": input_ext,
        "output_format": output_ext,
        "input_vertices": original_verts,
        "input_faces": original_faces,
        "output_vertices": len(mesh.vertices),
        "output_faces": len(mesh.faces),
        "output_size_bytes": output_size,
        "simplified": target_faces is not None and len(mesh.faces) < original_faces,
    }


def get_mesh_info(file_path: str) -> dict:
    """Get basic info about a mesh file without full processing."""
    mesh = trimesh.load(file_path, force="mesh")
    bounds = mesh.bounding_box.extents

    return {
        "format": Path(file_path).suffix.lstrip("."),
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "is_watertight": bool(mesh.is_watertight),
        "volume_cm3": round(mesh.volume / 1000, 2) if mesh.is_watertight else None,
        "surface_area_cm2": round(mesh.area / 100, 2),
        "bounding_box_mm": [round(x, 1) for x in bounds],
        "file_size_bytes": os.path.getsize(file_path),
    }
