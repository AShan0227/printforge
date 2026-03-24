"""Terms of Service for PrintForge."""

TOS_TEXT = """\
PrintForge Terms of Service
============================

Last updated: 2026-03-24

1. Content Ownership
   You retain full ownership of all images you upload and all 3D models
   generated from your content. PrintForge claims no rights over your
   input or output files.

2. Copyright Responsibility
   You are solely responsible for ensuring that the images you upload do
   not infringe on any third-party copyrights, trademarks, or other
   intellectual property rights. Do not upload images you do not have the
   right to use.

3. No Warranty
   PrintForge is provided "AS IS" without warranty of any kind, express
   or implied, including but not limited to warranties of merchantability,
   fitness for a particular purpose, and non-infringement.

4. Watertight Best-Effort
   PrintForge uses voxelization and marching cubes to produce watertight
   meshes. This process is best-effort; certain complex geometries may
   not result in a perfectly watertight output. Always verify your mesh
   in your slicer before printing.

5. Limitation of Liability
   In no event shall PrintForge or its contributors be liable for any
   direct, indirect, incidental, special, or consequential damages arising
   from the use of this software.

6. MIT License Base
   This software is released under the MIT License. You are free to use,
   copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the software, subject to the conditions above.

7. Changes
   We may update these terms at any time. Continued use of PrintForge
   after changes constitutes acceptance of the new terms.
"""


def get_tos() -> str:
    """Return the full Terms of Service text."""
    return TOS_TEXT
