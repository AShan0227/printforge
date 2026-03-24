"""Export formats: 3MF, STL, OBJ with proper metadata and settings."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {
    "3mf": {
        "extension": ".3mf",
        "mime_type": "application/vnd.ms-package.3dmanufacturing-3dmodel+xml",
        "description": "3D Manufacturing Format — includes print settings and metadata",
    },
    "stl": {
        "extension": ".stl",
        "mime_type": "application/sla",
        "description": "Stereolithography — binary mesh format, widely supported",
    },
    "obj": {
        "extension": ".obj",
        "mime_type": "text/plain",
        "description": "Wavefront OBJ — mesh with optional material file",
    },
}


def export_3mf(
    mesh,
    output_path: str,
    metadata: Optional[dict] = None,
) -> str:
    """Export mesh as 3MF with print settings metadata.

    Args:
        mesh: trimesh.Trimesh object.
        output_path: Path to save the .3mf file.
        metadata: Optional dict with keys like 'title', 'author', 'layer_height', etc.
    """
    output_path = str(Path(output_path).with_suffix(".3mf"))

    meta = metadata or {}

    # Set trimesh metadata for 3MF export
    if hasattr(mesh, "metadata"):
        mesh.metadata.update({
            "name": meta.get("title", "PrintForge Model"),
        })

    try:
        mesh.export(output_path, file_type="3mf")
        logger.info(f"Exported 3MF: {output_path}")
    except Exception as e:
        logger.warning(f"3MF export via trimesh failed ({e}), writing manually")
        _write_3mf_manual(mesh, output_path, meta)

    return output_path


def export_stl(mesh, output_path: str, binary: bool = True) -> str:
    """Export mesh as binary STL.

    Args:
        mesh: trimesh.Trimesh object.
        output_path: Path to save the .stl file.
        binary: If True (default), write binary STL. Otherwise ASCII.
    """
    output_path = str(Path(output_path).with_suffix(".stl"))

    if binary:
        mesh.export(output_path, file_type="stl")
    else:
        mesh.export(output_path, file_type="stl_ascii")

    logger.info(f"Exported STL: {output_path}")
    return output_path


def export_obj(mesh, output_path: str) -> str:
    """Export mesh as OBJ with MTL material file.

    Args:
        mesh: trimesh.Trimesh object.
        output_path: Path to save the .obj file.

    Returns:
        Path to the saved OBJ file.
    """
    output_path = str(Path(output_path).with_suffix(".obj"))

    mesh.export(output_path, file_type="obj")

    # Write a basic MTL file
    mtl_path = str(Path(output_path).with_suffix(".mtl"))
    mtl_name = Path(output_path).stem

    mtl_content = (
        f"# PrintForge Material\n"
        f"newmtl {mtl_name}_material\n"
        f"Ka 0.2 0.2 0.2\n"
        f"Kd 0.8 0.8 0.8\n"
        f"Ks 0.1 0.1 0.1\n"
        f"Ns 10.0\n"
        f"d 1.0\n"
    )

    Path(mtl_path).write_text(mtl_content)
    logger.info(f"Exported OBJ: {output_path} + {mtl_path}")

    return output_path


def _write_3mf_manual(mesh, output_path: str, metadata: dict):
    """Write a minimal 3MF file manually using zipfile."""
    import zipfile
    import struct

    vertices = mesh.vertices
    faces = mesh.faces

    # Build XML model
    title = metadata.get("title", "PrintForge Model")

    model_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <metadata name="Title">{title}</metadata>
  <metadata name="Application">PrintForge</metadata>
  <resources>
    <object id="1" type="model">
      <mesh>
        <vertices>
"""
    for v in vertices:
        model_xml += f'          <vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}" />\n'

    model_xml += "        </vertices>\n        <triangles>\n"

    for f in faces:
        model_xml += f'          <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}" />\n'

    model_xml += """        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1" />
  </build>
</model>"""

    content_types = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml" />
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />
</Relationships>"""

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("3D/3dmodel.model", model_xml)

    logger.info(f"Wrote 3MF manually: {output_path}")
