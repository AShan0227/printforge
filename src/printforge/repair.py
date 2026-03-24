"""STL Repair Tool: Fix broken meshes for 3D printing."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RepairReport:
    """Summary of repairs performed on a mesh."""
    input_faces: int
    output_faces: int
    input_vertices: int
    output_vertices: int
    was_watertight_before: bool
    is_watertight_after: bool
    repairs_performed: List[str] = field(default_factory=list)
    used_voxel_remesh: bool = False


class MeshRepair:
    """Take a broken STL/mesh and fix it for 3D printing.

    Repair strategy (in order):
      1. Remove degenerate faces (zero-area)
      2. Fix face winding / normals
      3. Fill holes
      4. Fix non-manifold edges (merge vertices)
      5. If still not watertight → voxelize + Marching Cubes remesh

    Args:
        voxel_resolution: Grid resolution for the voxel remesh fallback.
    """

    def __init__(self, voxel_resolution: int = 128):
        self.voxel_resolution = voxel_resolution

    def repair(self, mesh) -> tuple:
        """Repair a mesh and return (repaired_mesh, RepairReport).

        Args:
            mesh: A trimesh.Trimesh object.

        Returns:
            Tuple of (repaired trimesh.Trimesh, RepairReport).
        """
        import trimesh

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh).__name__}")

        repairs: List[str] = []
        input_faces = len(mesh.faces)
        input_vertices = len(mesh.vertices)
        was_watertight = bool(mesh.is_watertight)

        # Work on a copy so we don't mutate the original
        mesh = mesh.copy()

        # 1. Remove degenerate faces (zero-area triangles)
        degen_before = len(mesh.faces)
        nondegenerate = trimesh.triangles.area(mesh.triangles) > 0
        if not nondegenerate.all():
            mesh.update_faces(nondegenerate)
            removed = degen_before - len(mesh.faces)
            repairs.append(f"Removed {removed} degenerate faces")

        # 2. Merge close vertices (fixes non-manifold from duplicate verts)
        verts_before = len(mesh.vertices)
        mesh.merge_vertices()
        if len(mesh.vertices) < verts_before:
            merged = verts_before - len(mesh.vertices)
            repairs.append(f"Merged {merged} duplicate vertices")

        # 3. Fix normals and winding
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)
        repairs.append("Fixed normals and winding")

        # 4. Fill holes
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            repairs.append("Filled holes")

        # 5. Fix non-manifold edges by removing problematic faces
        if not mesh.is_watertight:
            self._fix_non_manifold(mesh, repairs)

        # 6. Voxel remesh fallback for severe cases
        used_voxel = False
        if not mesh.is_watertight:
            logger.info("Standard repairs insufficient — applying voxel remesh")
            mesh = self._voxel_remesh(mesh)
            repairs.append(f"Voxel remesh at resolution {self.voxel_resolution}")
            used_voxel = True

        report = RepairReport(
            input_faces=input_faces,
            output_faces=len(mesh.faces),
            input_vertices=input_vertices,
            output_vertices=len(mesh.vertices),
            was_watertight_before=was_watertight,
            is_watertight_after=bool(mesh.is_watertight),
            repairs_performed=repairs,
            used_voxel_remesh=used_voxel,
        )

        logger.info(
            "Repair done: %d→%d faces, watertight=%s, repairs=%s",
            input_faces,
            len(mesh.faces),
            report.is_watertight_after,
            repairs,
        )

        return mesh, report

    def _fix_non_manifold(self, mesh, repairs: List[str]):
        """Attempt to remove non-manifold faces."""
        import trimesh

        try:
            # Identify edges shared by more than 2 faces
            edges = mesh.edges_sorted
            unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
            non_manifold = unique_edges[edge_counts > 2]

            if len(non_manifold) > 0:
                # Find faces that reference non-manifold edges
                bad_face_indices = set()
                edge_set = {tuple(e) for e in non_manifold}
                for i, face in enumerate(mesh.faces):
                    face_edges = [
                        tuple(sorted([face[0], face[1]])),
                        tuple(sorted([face[1], face[2]])),
                        tuple(sorted([face[2], face[0]])),
                    ]
                    if any(e in edge_set for e in face_edges):
                        bad_face_indices.add(i)

                if bad_face_indices:
                    keep = [i for i in range(len(mesh.faces)) if i not in bad_face_indices]
                    mesh.faces = mesh.faces[keep]
                    mesh.remove_unreferenced_vertices()
                    repairs.append(f"Removed {len(bad_face_indices)} non-manifold faces")

                    # Try filling holes again
                    trimesh.repair.fix_normals(mesh)
                    trimesh.repair.fill_holes(mesh)
        except Exception as e:
            logger.warning("Non-manifold fix failed: %s", e)

    def _voxel_remesh(self, mesh):
        """Voxelize the mesh and extract a watertight surface via Marching Cubes."""
        pitch = mesh.bounding_box.extents.max() / self.voxel_resolution
        voxel_grid = mesh.voxelized(pitch).fill()
        return voxel_grid.marching_cubes
