# Pipeline patch — add Stage 3.5: Tail detection & removal
# Apply by importing printforge.tail_remover and calling remove_tail_after_watertight
#
# In pipeline.py, after:
#   watertight_mesh = self._make_watertight(raw_mesh)
# Add:
#   watertight_mesh, tail_result, tail_warnings = _remove_tail_after_watertight(
#       self, watertight_mesh
#   )

import logging
import trimesh

from .tail_remover import remove_tail, TailRemovalResult

logger = logging.getLogger(__name__)


def _remove_tail_after_watertight(pipeline, mesh):
    """
    Hook into PrintForgePipeline: call remove_tail after _make_watertight.

    Returns (cleaned_mesh, tail_result, warnings).
    """
    warnings = []

    try:
        result = remove_tail(mesh)
    except Exception as e:
        logger.warning(f"Tail removal failed: {e}")
        return mesh, None, []

    if result.tail_detected:
        tail_warn = (
            f"Stage 3.5 — Tail detected and removed along "
            f"{result.tail_direction}: "
            f"{result.removed_percentage*100:.1f}% vertices removed"
        )
        logger.warning(tail_warn)
        warnings.append(tail_warn)

        # Rebuild mesh with cleaned vertices
        cleaned = trimesh.Trimesh(
            vertices=result.cleaned_verts,
            faces=mesh.faces,
        )
        return cleaned, result, warnings

    return mesh, result, warnings
