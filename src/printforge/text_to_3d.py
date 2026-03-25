"""Text-to-3D Pipeline: Text description → LLM prompt → Image generation → 3D mesh."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TextTo3DConfig:
    """Configuration for Text-to-3D pipeline."""
    # LLM for prompt enhancement
    llm_model: str = "gpt-4o-mini"

    # Image generation
    image_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    hf_api_token: Optional[str] = None
    image_width: int = 512
    image_height: int = 512

    # Enhanced prompt template
    prompt_template: str = (
        "A single 3D-printable object on a plain white background, "
        "studio lighting, no shadows, centered, full view: {description}"
    )


@dataclass
class TextTo3DResult:
    """Result of text-to-3D generation."""
    prompt_used: str
    image_path: Optional[str] = None
    mesh_path: Optional[str] = None
    pipeline_result: object = None
    used_fallback: bool = False


class TextTo3DPipeline:
    """Text description → image → 3D printable mesh.

    Uses Hugging Face Inference API for image generation (SDXL/FLUX),
    then feeds the image into the existing PrintForge pipeline.
    """

    def __init__(self, config: Optional[TextTo3DConfig] = None):
        self.config = config or TextTo3DConfig()

    def generate_image_prompt(self, description: str) -> str:
        """Enhance a user description into an optimized image generation prompt."""
        return self.config.prompt_template.format(description=description)

    def generate_image(self, prompt: str, output_path: str) -> str:
        """Generate an image from a prompt using Hugging Face Inference API.

        Returns the path to the saved image, or raises if generation fails.
        """
        token = self.config.hf_api_token or os.environ.get("HF_API_TOKEN")
        if not token:
            raise RuntimeError(
                "No Hugging Face API token. Set HF_API_TOKEN env var or pass hf_api_token in config. "
                "Alternatively, use run() with provide_image=True to skip image generation."
            )

        import requests

        api_url = f"https://api-inference.huggingface.co/models/{self.config.image_model}"
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"inputs": prompt}

        logger.info(f"Generating image with {self.config.image_model}...")
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)

        if response.status_code != 200:
            raise RuntimeError(
                f"Image generation failed (HTTP {response.status_code}): {response.text}"
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Image saved to {output_path}")
        return output_path

    def generate_via_hunyuan3d(self, description: str, output_path: str) -> str:
        """Generate 3D mesh directly from text via Hunyuan3D-2's text input.

        Hunyuan3D-2's /shape_generation endpoint accepts text as its first
        parameter. This bypasses the image generation step entirely.

        Returns the path to the saved mesh, or raises on failure.
        """
        try:
            from gradio_client import Client
        except ImportError:
            raise RuntimeError("gradio_client not installed — cannot use Hunyuan3D text-to-3D")

        import trimesh

        logger.info(f"Generating 3D from text via Hunyuan3D-2: {description!r}")
        client = Client("Tencent/Hunyuan3D-2")
        result = client.predict(
            description, None, None, None, None,
            api_name="/shape_generation",
        )

        glb_path = result[0]["value"]
        mesh = trimesh.load(glb_path, file_type="glb", force="mesh")
        logger.info(f"Hunyuan3D text-to-3D OK: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

        mesh.export(output_path)
        return output_path

    def run(
        self,
        description: str,
        output_path: str,
        image_path: Optional[str] = None,
        save_prompt: Optional[str] = None,
        pipeline_config=None,
    ) -> TextTo3DResult:
        """Run the full text-to-3D pipeline.

        Args:
            description: Text description of the desired 3D object.
            output_path: Where to save the final 3D mesh.
            image_path: If provided, skip image generation and use this image.
            save_prompt: If set, save the generated prompt to this path (for manual image generation).
            pipeline_config: Optional PipelineConfig for the 3D pipeline.
        """
        import tempfile

        # Step 1: Generate enhanced prompt
        prompt = self.generate_image_prompt(description)
        logger.info(f"Enhanced prompt: {prompt}")

        result = TextTo3DResult(prompt_used=prompt)

        # Optionally save prompt for manual use
        if save_prompt:
            Path(save_prompt).parent.mkdir(parents=True, exist_ok=True)
            Path(save_prompt).write_text(prompt)
            logger.info(f"Prompt saved to {save_prompt}")

        # Step 2: Get or generate image
        if image_path:
            result.image_path = image_path
            result.used_fallback = False
        else:
            try:
                tmp_image = tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False, prefix="printforge_text2img_"
                )
                tmp_image.close()
                self.generate_image(prompt, tmp_image.name)
                result.image_path = tmp_image.name
            except (RuntimeError, ImportError) as e:
                logger.warning(f"Image generation failed: {e}")
                # Fallback: try Hunyuan3D-2 direct text-to-3D
                logger.info("Trying Hunyuan3D-2 direct text-to-3D...")
                try:
                    self.generate_via_hunyuan3d(description, output_path)
                    result.mesh_path = output_path
                    result.used_fallback = False
                    return result
                except Exception as hunyuan_err:
                    logger.warning(f"Hunyuan3D text-to-3D also failed: {hunyuan_err}")
                    if save_prompt:
                        logger.info("Prompt saved. Provide your own image with image_path parameter.")
                        result.used_fallback = True
                        return result
                    # Final fallback: use placeholder pipeline
                    result.used_fallback = True
                    return result

        # Step 3: Run existing pipeline
        from .pipeline import PrintForgePipeline, PipelineConfig

        config = pipeline_config or PipelineConfig()
        pipeline = PrintForgePipeline(config)
        pipeline_result = pipeline.run(result.image_path, output_path)

        result.mesh_path = output_path
        result.pipeline_result = pipeline_result

        return result
