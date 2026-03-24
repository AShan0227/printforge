"""PrintForge Web API — Upload image, get 3D print file."""

import tempfile
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .pipeline import PrintForgePipeline, PipelineConfig

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PrintForge",
    description="One photo to 3D print",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (loaded once)
_pipeline: PrintForgePipeline | None = None


def get_pipeline() -> PrintForgePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PrintForgePipeline(PipelineConfig())
    return _pipeline


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    format: str = "3mf",
    size_mm: float = 50.0,
    add_base: bool = False,
):
    """Upload an image, get a 3D printable file."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (jpg/png)")
    
    # Save uploaded image to temp file
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        content = await image.read()
        tmp_in.write(content)
        input_path = tmp_in.name
    
    # Generate output path
    output_suffix = ".3mf" if format == "3mf" else ".stl"
    with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as tmp_out:
        output_path = tmp_out.name
    
    # Run pipeline
    try:
        pipeline = get_pipeline()
        pipeline.config.output_format = format
        pipeline.config.scale_mm = size_mm
        pipeline.config.add_base = add_base
        
        result = pipeline.run(input_path, output_path)
        
        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"printforge_output{output_suffix}",
            headers={
                "X-PrintForge-Vertices": str(result.vertices),
                "X-PrintForge-Faces": str(result.faces),
                "X-PrintForge-Watertight": str(result.is_watertight),
                "X-PrintForge-Duration-Ms": str(int(result.duration_ms)),
            },
        )
    except Exception as e:
        logger.exception("Pipeline failed")
        raise HTTPException(500, f"Generation failed: {str(e)}")


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)):
    """Analyze an image and return metadata without generating the full mesh."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")
    
    return JSONResponse({
        "status": "ok",
        "message": "Analysis endpoint — coming soon",
        "supported_formats": ["3mf", "stl"],
        "default_size_mm": 50.0,
    })


def start():
    """Start the web server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
