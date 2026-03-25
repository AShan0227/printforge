#!/usr/bin/env python3
"""Generate example STL models using trimesh.creation."""

import os
import trimesh

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def main():
    os.makedirs(EXAMPLES_DIR, exist_ok=True)

    # Cube 50mm
    cube = trimesh.creation.box(extents=[50.0, 50.0, 50.0])
    cube.export(os.path.join(EXAMPLES_DIR, "cube_50mm.stl"), file_type="stl")
    print(f"Created cube_50mm.stl  ({len(cube.vertices)} verts, {len(cube.faces)} faces)")

    # Sphere 40mm diameter
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
    sphere.export(os.path.join(EXAMPLES_DIR, "sphere_40mm.stl"), file_type="stl")
    print(f"Created sphere_40mm.stl ({len(sphere.vertices)} verts, {len(sphere.faces)} faces)")

    # Cylinder 30mm diameter, 30mm height
    cylinder = trimesh.creation.cylinder(radius=15.0, height=30.0, sections=64)
    cylinder.export(os.path.join(EXAMPLES_DIR, "cylinder_30mm.stl"), file_type="stl")
    print(f"Created cylinder_30mm.stl ({len(cylinder.vertices)} verts, {len(cylinder.faces)} faces)")

    print(f"\nAll examples saved to {EXAMPLES_DIR}/")


if __name__ == "__main__":
    main()
