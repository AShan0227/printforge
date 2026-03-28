"""Export trimesh meshes to GLB (glTF Binary) format with basic material color."""

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def export_glb(
    mesh,
    output_path: str,
    color: Optional[str] = None,
) -> str:
    """Export a trimesh.Trimesh to GLB format with optional base color.

    Args:
        mesh: trimesh.Trimesh object to export.
        output_path: Path to save the .glb file.
        color: Optional hex color string, e.g. "#FF5733".
               If None, uses a neutral gray (0.8, 0.8, 0.8).

    Returns:
        Path to the saved GLB file.
    """
    import numpy as np
    import trimesh

    output_path = str(Path(output_path).with_suffix(".glb"))

    # Ensure mesh is a Trimesh
    if not isinstance(mesh, trimesh.Trimesh):
        if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        else:
            raise ValueError("Cannot convert mesh to trimesh.Trimesh")

    # Fix normals before export
    trimesh.repair.fix_normals(mesh)

    # Only apply color if mesh has no existing texture/material
    has_texture = (
        hasattr(mesh, 'visual') and mesh.visual is not None
        and hasattr(mesh.visual, 'kind') and mesh.visual.kind == 'texture'
    )
    has_vertex_colors = (
        hasattr(mesh, 'visual') and mesh.visual is not None
        and hasattr(mesh.visual, 'vertex_colors')
        and mesh.visual.vertex_colors is not None
        and len(mesh.visual.vertex_colors) > 0
    )

    if has_texture or has_vertex_colors:
        # Preserve existing visual (PBR texture from Tripo, vertex colors, etc.)
        logger.info("Preserving existing mesh visual/texture for GLB export")
    elif color is not None:
        rgba = _hex_to_rgba(color)
        _apply_vertex_color(mesh, rgba)
    else:
        # Default neutral gray only when no visual exists
        _apply_vertex_color(mesh, (0.8, 0.8, 0.8, 1.0))

    mesh.export(output_path, file_type="glb")
    logger.info(f"Exported GLB: {output_path} (color={color or 'default gray'})")
    return output_path


def _hex_to_rgba(hex_color: str) -> Tuple[float, float, float, float]:
    """Convert hex color string to RGBA tuple (0-1 range).

    Args:
        hex_color: Hex string like "#FF5733" or "FF5733".

    Returns:
        RGBA tuple with values in [0, 1].
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 6:
        hex_color += "FF"  # Add full opacity if alpha missing

    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    a = int(hex_color[6:8], 16) / 255.0
    return (r, g, b, a)


def _apply_vertex_color(mesh, rgba: Tuple[float, float, float, float]) -> None:
    """Apply a uniform RGBA color to all vertices of a mesh.

    Args:
        mesh: trimesh.Trimesh object.
        rgba: RGBA tuple with values in [0, 1].
    """
    import numpy as np
    import trimesh as _trimesh

    n_verts = len(mesh.vertices)
    color_data = np.array(rgba, dtype=np.float32)
    mesh.visual = _trimesh.visual.ColorVisuals()
    mesh.visual.vertex_colors = np.tile(color_data, (n_verts, 1))
