"""
Tail Detection & Removal for 3D Meshes
========================================
Detects anomalous elongated extensions ("tails") from TripoSR/Hunyuan3D
reconstructions and removes them via plane-cut post-processing.
"""

from __future__ import annotations

import logging
import numpy as np
import trimesh
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class TailRemovalResult:
    original_verts: np.ndarray
    cleaned_verts: np.ndarray
    tail_detected: bool
    tail_direction: str | None
    removed_percentage: float
    tail_axis_idx: int | None = None
    tail_sign: int | None = None

    def __post_init__(self):
        self.removed_percentage = max(0.0, min(1.0, self.removed_percentage))


# -------------------------------------------------------------------------- #
# Internal helpers
# -------------------------------------------------------------------------- #

def _axis_name(idx: int, sign: int) -> str:
    axis = ("x", "y", "z")[idx]
    return f"{'+' if sign >= 0 else '-'}{axis}"


def _submesh(mesh: trimesh.Trimesh, vertex_indices: List[int]) -> trimesh.Trimesh:
    """Call submesh, handling trimesh version differences (list vs mesh return)."""
    result = mesh.submesh([vertex_indices], append=False)
    if isinstance(result, list):
        return result[0]
    return result


def _connected_components(mesh: trimesh.Trimesh) -> List[np.ndarray]:
    adj = mesh.vertex_adjacency_graph
    n = len(mesh.vertices)
    visited = np.zeros(n, dtype=bool)
    components: List[np.ndarray] = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        group = []
        while stack:
            v = stack.pop()
            if visited[v]:
                continue
            visited[v] = True
            group.append(v)
            for nb in adj[v]:
                if not visited[nb]:
                    stack.append(nb)
        if group:
            components.append(np.array(group, dtype=int))
    return components


def _detect_tail_axis(extents: np.ndarray, threshold: float = 2.5):
    avg_other = (extents.sum() - extents) / 2.0
    for idx in range(3):
        ratio = extents[idx] / max(avg_other[idx], 1e-9)
        if ratio >= threshold:
            return idx
    return None


def _find_cut_plane(mesh, main_idx, tail_axis, tail_sign):
    axis_coords = mesh.vertices[main_idx, tail_axis]
    offset = axis_coords.mean() + tail_sign * 2.0 * axis_coords.std()
    normal = np.zeros(3)
    normal[tail_axis] = 1.0
    return normal, offset


def _no_tail(original_verts):
    return TailRemovalResult(
        original_verts=original_verts,
        cleaned_verts=original_verts,
        tail_detected=False,
        tail_direction=None,
        removed_percentage=0.0,
    )


# -------------------------------------------------------------------------- #
# Public API
# -------------------------------------------------------------------------- #

def remove_tail(
    mesh: trimesh.Trimesh,
    extent_ratio_threshold: float = 2.5,
    min_tail_vertices_ratio: float = 0.01,
    max_aspect_ratio: float = 12.0,
) -> TailRemovalResult:
    """
    Detect and remove an elongated tail from a 3D mesh.

    Detection logic:
      1. Find largest connected component.
      2. If its bounding-box extent ratio exceeds *extent_ratio_threshold*,
         flag the elongated axis as the tail direction.
      3. Skip if the mesh's overall aspect ratio > *max_aspect_ratio*
         (i.e., it is a legitimate elongated object like a pencil).
      4. Slice with a plane at mean + 2σ along the tail axis.
      5. Fill holes and fix normals.
    """
    original_verts = mesh.vertices.copy()
    original_n = len(mesh.vertices)

    if original_n == 0:
        return _no_tail(original_verts)

    # Step 1 — connected components
    components = _connected_components(mesh)
    if not components:
        return _no_tail(original_verts)

    main_idx = components[np.argmax([len(c) for c in components])]
    main_mesh = _submesh(mesh, main_idx.tolist())

    # Step 2 — bounding box analysis
    extents = main_mesh.bounding_box.extents
    aspect = extents.max() / max(extents.min(), 1e-9)

    if aspect >= max_aspect_ratio:
        logger.info(
            f"Tail removal skipped: aspect ratio {aspect:.1f} >= {max_aspect_ratio} "
            "(likely a legitimate elongated object)."
        )
        return _no_tail(original_verts)

    tail_axis = _detect_tail_axis(extents, extent_ratio_threshold)
    if tail_axis is None:
        return _no_tail(original_verts)

    # Step 3 — tail sign (positive or negative direction)
    centroid = main_mesh.vertices.mean(axis=0)
    bounds = main_mesh.bounds
    dist_pos = abs(bounds[1, tail_axis] - centroid[tail_axis])
    dist_neg = abs(centroid[tail_axis] - bounds[0, tail_axis])
    tail_sign = 1 if dist_pos >= dist_neg else -1
    tail_dir = _axis_name(tail_axis, tail_sign)

    logger.info(
        f"Tail detected along {tail_dir}: extents={extents}, aspect={aspect:.1f}"
    )

    # Step 4 — plane cut
    normal, offset = _find_cut_plane(mesh, main_idx, tail_axis, tail_sign)
    try:
        pieces = mesh.slice_plane(normal, offset, cap=True)
        if pieces is None or len(pieces) == 0:
            raise ValueError("slice_plane returned empty")
        cleaned = max(pieces, key=lambda p: len(p.vertices))
    except Exception as e:
        logger.warning(f"Plane cut failed ({e}), returning original mesh.")
        return _no_tail(original_verts)

    # Step 5 — watertight repair
    trimesh.repair.fill_holes(cleaned)
    trimesh.repair.fix_normals(cleaned)
    trimesh.repair.fix_winding(cleaned)

    # Step 6 — sanity check
    removed_ratio = 1.0 - (len(cleaned.vertices) / original_n)
    if removed_ratio < min_tail_vertices_ratio:
        logger.info(
            f"Tail removal reverted: only {removed_ratio*100:.1f}% removed "
            f"(< min_tail_vertices_ratio={min_tail_vertices_ratio})."
        )
        return _no_tail(original_verts)

    logger.info(
        f"Tail removed: {removed_ratio*100:.1f}% vertices, "
        f"new mesh {len(cleaned.vertices)} verts / {len(cleaned.faces)} faces"
    )

    return TailRemovalResult(
        original_verts=original_verts,
        cleaned_verts=cleaned.vertices.copy(),
        tail_detected=True,
        tail_direction=tail_dir,
        removed_percentage=removed_ratio,
        tail_axis_idx=tail_axis,
        tail_sign=tail_sign,
    )


def cleaned_mesh(result: TailRemovalResult, faces: np.ndarray) -> trimesh.Trimesh:
    """Reconstruct a Trimesh from a TailRemovalResult."""
    verts = result.cleaned_verts if result.tail_detected else result.original_verts
    return trimesh.Trimesh(vertices=verts, faces=faces)
