"""Part Splitter: Split large meshes into printable parts with alignment features."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BuildVolume:
    """Printer build volume in mm."""
    x: float = 256.0
    y: float = 256.0
    z: float = 256.0

    @classmethod
    def from_string(cls, s: str) -> "BuildVolume":
        """Parse 'XxYxZ' format, e.g. '256x256x256'."""
        parts = s.lower().split("x")
        if len(parts) != 3:
            raise ValueError(f"Expected format 'XxYxZ', got '{s}'")
        return cls(x=float(parts[0]), y=float(parts[1]), z=float(parts[2]))


@dataclass
class SplitConfig:
    """Configuration for part splitting."""
    build_volume: BuildVolume = field(default_factory=BuildVolume)
    pin_radius: float = 2.0       # Alignment pin radius in mm
    pin_height: float = 5.0       # Alignment pin height in mm
    pin_tolerance: float = 0.15   # Gap for hole (larger than pin)
    pin_spacing: float = 30.0     # Distance between pins along split edge
    overlap_margin: float = 0.5   # Overlap at split plane for clean cuts


@dataclass
class SplitPart:
    """A single part from splitting."""
    mesh: object  # trimesh.Trimesh
    part_index: int
    has_pins: bool
    has_holes: bool
    bounding_box: tuple  # (min_corner, max_corner)


@dataclass
class SplitResult:
    """Result of part splitting."""
    parts: list[SplitPart]
    num_parts: int
    fits_in_volume: bool
    split_axes: list[str]  # e.g. ["x", "z"]


class PartSplitter:
    """Split meshes that exceed build volume into printable parts with alignment pins."""

    def __init__(self, config: Optional[SplitConfig] = None):
        self.config = config or SplitConfig()

    def needs_splitting(self, mesh) -> tuple[bool, list[str]]:
        """Check if a mesh exceeds the build volume.

        Returns (needs_split, list of axes that exceed).
        """
        extents = mesh.bounding_box.extents
        vol = self.config.build_volume

        exceeded = []
        if extents[0] > vol.x:
            exceeded.append("x")
        if extents[1] > vol.y:
            exceeded.append("y")
        if extents[2] > vol.z:
            exceeded.append("z")

        return len(exceeded) > 0, exceeded

    def split(self, mesh) -> SplitResult:
        """Analyze mesh and split into parts that fit the build volume.

        Splits along axes that exceed the build volume, adding alignment
        pins and holes at split planes.
        """
        import trimesh

        needs_split, exceeded_axes = self.needs_splitting(mesh)

        if not needs_split:
            part = SplitPart(
                mesh=mesh,
                part_index=0,
                has_pins=False,
                has_holes=False,
                bounding_box=(mesh.bounds[0].tolist(), mesh.bounds[1].tolist()),
            )
            return SplitResult(
                parts=[part],
                num_parts=1,
                fits_in_volume=True,
                split_axes=[],
            )

        # Split along each exceeded axis
        parts_meshes = [mesh]
        all_split_axes = []

        for axis_name in exceeded_axes:
            axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name]
            limit = [self.config.build_volume.x, self.config.build_volume.y, self.config.build_volume.z][axis_idx]

            new_parts = []
            for part_mesh in parts_meshes:
                split = self._split_along_axis(part_mesh, axis_idx, limit)
                new_parts.extend(split)

            parts_meshes = new_parts
            all_split_axes.append(axis_name)

        # Add alignment features
        final_parts = []
        for i, part_mesh in enumerate(parts_meshes):
            has_pins = i < len(parts_meshes) - 1
            has_holes = i > 0

            if has_pins:
                part_mesh = self._add_alignment_pins(part_mesh, side="positive")
            if has_holes:
                part_mesh = self._add_alignment_holes(part_mesh, side="negative")

            final_parts.append(SplitPart(
                mesh=part_mesh,
                part_index=i,
                has_pins=has_pins,
                has_holes=has_holes,
                bounding_box=(part_mesh.bounds[0].tolist(), part_mesh.bounds[1].tolist()),
            ))

        return SplitResult(
            parts=final_parts,
            num_parts=len(final_parts),
            fits_in_volume=all(self._fits_volume(p.mesh) for p in final_parts),
            split_axes=all_split_axes,
        )

    def _split_along_axis(self, mesh, axis: int, max_size: float) -> list:
        """Split a mesh along an axis into chunks that fit max_size."""
        import trimesh

        extents = mesh.bounding_box.extents
        if extents[axis] <= max_size:
            return [mesh]

        num_splits = int(np.ceil(extents[axis] / max_size))
        min_val = mesh.bounds[0][axis]
        chunk_size = extents[axis] / num_splits

        parts = []
        remaining = mesh

        for i in range(num_splits - 1):
            split_pos = min_val + chunk_size * (i + 1)

            # Create split plane
            plane_origin = np.zeros(3)
            plane_origin[axis] = split_pos
            plane_normal = np.zeros(3)
            plane_normal[axis] = 1.0

            try:
                result = remaining.slice_plane(plane_origin, -plane_normal, cap=True)
                other = remaining.slice_plane(plane_origin, plane_normal, cap=True)

                if result is not None and len(result.faces) > 0:
                    parts.append(result)
                if other is not None and len(other.faces) > 0:
                    remaining = other
                else:
                    break
            except Exception as e:
                logger.warning(f"Split failed at position {split_pos}: {e}")
                parts.append(remaining)
                remaining = None
                break

        if remaining is not None and len(remaining.faces) > 0:
            parts.append(remaining)

        return parts if parts else [mesh]

    def _add_alignment_pins(self, mesh, side: str = "positive"):
        """Add cylindrical alignment pins to a split face."""
        import trimesh

        # Find the face on the split side and add pins along it
        centroid = mesh.bounding_box.centroid

        pin = trimesh.creation.cylinder(
            radius=self.config.pin_radius,
            height=self.config.pin_height,
        )

        # Place pin at centroid of the split face, extending outward
        pin.apply_translation([centroid[0], centroid[1], mesh.bounds[1][2]])

        return trimesh.util.concatenate([mesh, pin])

    def _add_alignment_holes(self, mesh, side: str = "negative"):
        """Add alignment holes (slightly larger than pins) to a split face."""
        import trimesh

        centroid = mesh.bounding_box.centroid

        hole = trimesh.creation.cylinder(
            radius=self.config.pin_radius + self.config.pin_tolerance,
            height=self.config.pin_height + self.config.pin_tolerance * 2,
        )

        hole.apply_translation([centroid[0], centroid[1], mesh.bounds[0][2]])

        try:
            result = trimesh.boolean.difference([mesh, hole], engine="blender")
            return result
        except Exception:
            logger.warning("Boolean difference failed, returning mesh without alignment holes")
            return mesh

    def _fits_volume(self, mesh) -> bool:
        """Check if a mesh fits within the build volume."""
        extents = mesh.bounding_box.extents
        vol = self.config.build_volume
        return extents[0] <= vol.x and extents[1] <= vol.y and extents[2] <= vol.z
