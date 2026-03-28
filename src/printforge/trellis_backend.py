"""
TRELLIS Backend — High-quality image-to-3D via TRELLIS (CVPR 2025).
===================================================================

Uses the JeffreyXiang/TRELLIS HuggingFace Space via gradio_client.
TRELLIS produces significantly better 3D meshes than Hunyuan3D for single images
because it uses Structured Latent Representation (SLAT) with 2B parameters.

Pipeline: preprocess_image → image_to_3d → extract_glb → trimesh load

Advantages over Hunyuan3D:
  - No fake multi-view synthesis needed (single image direct to 3D)
  - Better geometric consistency
  - Supports Mesh/NeRF/GS output formats
  - Less prone to "tail" artifacts
"""

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrellisConfig:
    """Configuration for TRELLIS inference."""
    space_id: str = "JeffreyXiang/TRELLIS"
    seed: int = 0
    randomize_seed: bool = True
    # Structure sampling
    ss_guidance_strength: float = 7.5
    ss_sampling_steps: int = 12
    # Structured latent sampling
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 12
    # GLB extraction
    mesh_simplify: float = 0.95
    texture_size: int = 1024
    # Timeout
    timeout_seconds: int = 300


@dataclass
class TrellisResult:
    """Result from TRELLIS inference."""
    mesh: object  # trimesh.Trimesh
    glb_path: str
    vertices: int
    faces: int
    preview_video_path: Optional[str] = None


def _get_hf_token() -> Optional[str]:
    """Get HuggingFace token from env or file."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".openclaw" / "workspace" / ".hf_token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


class TrellisBackend:
    """Image-to-3D via TRELLIS HuggingFace Space.

    Usage:
        backend = TrellisBackend()
        result = backend.generate(image_path_or_pil)
        mesh = result.mesh  # trimesh.Trimesh
    """

    def __init__(self, config: Optional[TrellisConfig] = None):
        self.config = config or TrellisConfig()
        self._client = None

    def _get_client(self):
        """Lazy-init Gradio client."""
        if self._client is None:
            from gradio_client import Client

            hf_token = _get_hf_token()
            kwargs = {"verbose": False}
            if hf_token:
                kwargs["token"] = hf_token
                logger.info("TRELLIS: Using authenticated HF access")
            else:
                logger.info("TRELLIS: Using anonymous access (may have quota limits)")

            self._client = Client(self.config.space_id, **kwargs)
        return self._client

    def generate(self, image, save_glb_path: Optional[str] = None) -> Optional[TrellisResult]:
        """Generate 3D mesh from a single image.

        Args:
            image: Path string or PIL.Image
            save_glb_path: Optional path to save the GLB file

        Returns:
            TrellisResult with mesh and metadata, or None on failure
        """
        try:
            import trimesh
            from PIL import Image as PILImage
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return None

        temp_files = []

        try:
            # Prepare image path
            if isinstance(image, str) or isinstance(image, Path):
                image_path = str(image)
            else:
                # PIL Image — save to temp
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                image.save(tmp, format="PNG")
                tmp.close()
                image_path = tmp.name
                temp_files.append(tmp.name)

            client = self._get_client()

            # Step 1: Start session
            logger.info("TRELLIS: Starting session...")
            try:
                client.predict(api_name="/start_session")
            except Exception as e:
                logger.warning(f"TRELLIS: start_session warning (may be OK): {e}")

            # Step 2: Preprocess image
            logger.info("TRELLIS: Preprocessing image...")
            preprocessed = client.predict(
                image=image_path,
                api_name="/preprocess_image",
            )
            logger.info(f"TRELLIS: Preprocessed image: {preprocessed}")

            # Step 3: Get seed
            seed = client.predict(
                randomize_seed=self.config.randomize_seed,
                seed=self.config.seed,
                api_name="/get_seed",
            )
            logger.info(f"TRELLIS: Using seed: {seed}")

            # Step 4: Image to 3D
            logger.info("TRELLIS: Generating 3D asset (this may take 30-120s)...")
            result_3d = client.predict(
                image=preprocessed,
                multiimages=[],
                seed=seed,
                ss_guidance_strength=self.config.ss_guidance_strength,
                ss_sampling_steps=self.config.ss_sampling_steps,
                slat_guidance_strength=self.config.slat_guidance_strength,
                slat_sampling_steps=self.config.slat_sampling_steps,
                multiimage_algo="stochastic",
                api_name="/image_to_3d",
            )
            logger.info(f"TRELLIS: 3D generation complete")

            # result_3d is a dict with 'video' key for preview
            preview_video = None
            if isinstance(result_3d, dict) and "video" in result_3d:
                preview_video = result_3d["video"]

            # Step 5: Extract GLB
            logger.info("TRELLIS: Extracting GLB mesh...")
            glb_result = client.predict(
                mesh_simplify=self.config.mesh_simplify,
                texture_size=self.config.texture_size,
                api_name="/extract_glb",
            )

            # glb_result is (model_path, download_path)
            if isinstance(glb_result, (list, tuple)):
                glb_path = glb_result[1] if len(glb_result) > 1 else glb_result[0]
            else:
                glb_path = glb_result

            if not glb_path or not Path(glb_path).exists():
                logger.error(f"TRELLIS: GLB file not found at {glb_path}")
                return None

            # Step 6: Load mesh
            mesh = trimesh.load(glb_path, file_type="glb", force="mesh")
            logger.info(
                f"TRELLIS: Loaded mesh — {len(mesh.vertices)} verts, "
                f"{len(mesh.faces)} faces, watertight={mesh.is_watertight}"
            )

            # Save GLB if requested
            final_glb = glb_path
            if save_glb_path:
                shutil.copy2(glb_path, save_glb_path)
                final_glb = save_glb_path

            return TrellisResult(
                mesh=mesh,
                glb_path=final_glb,
                vertices=len(mesh.vertices),
                faces=len(mesh.faces),
                preview_video_path=preview_video,
            )

        except Exception as e:
            logger.error(f"TRELLIS inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            for f in temp_files:
                try:
                    os.unlink(f)
                except OSError:
                    pass

    def is_available(self) -> bool:
        """Check if TRELLIS Space is reachable."""
        try:
            self._get_client()
            return True
        except Exception:
            return False
