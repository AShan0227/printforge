"""PrintForge Web API — Upload image, get 3D print file."""

import asyncio
import json
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipeline import PrintForgePipeline, PipelineConfig
from .print_optimizer import PrintOptimizer, PRINTER_PRESETS
from .formats import SUPPORTED_FORMATS
from .cache import ImageCache

logger = logging.getLogger(__name__)


# ── Pydantic Response Models ──────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    version: str = Field(..., json_schema_extra={"example": "1.1.0"})


class AnalyzeResponse(BaseModel):
    status: str
    message: str
    supported_formats: List[str]
    default_size_mm: float


class TextTo3DFallbackResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "fallback"})
    prompt: str
    message: str


class OrientationInfo(BaseModel):
    height_mm: float
    support_estimate_mm3: float
    base_area_mm2: float
    score: float
    overhang_percentage: float
    support_contact_area_mm2: float


class MaterialEstimate(BaseModel):
    print_time_minutes: int
    filament_grams: float
    filament_meters: float
    layer_count: int


class PrintIssue(BaseModel):
    severity: str
    category: str
    message: str


class MeshInfo(BaseModel):
    vertices: int
    faces: int
    is_watertight: bool
    bounding_box_mm: List[float]


class OptimizeResponse(BaseModel):
    status: str
    printer: str
    orientation: OrientationInfo
    estimate: MaterialEstimate
    issues: List[PrintIssue]
    mesh_info: MeshInfo


class CostBreakdown(BaseModel):
    filament_grams: float
    filament_meters: float
    filament_cost_usd: float
    print_time_hours: float
    electricity_cost_usd: float
    total_cost_usd: float


class CostResponse(BaseModel):
    status: str
    material: str
    infill_percent: float
    layer_height_mm: float
    cost: CostBreakdown
    mesh_info: MeshInfo


class SplitPartInfo(BaseModel):
    index: int
    has_pins: bool
    has_holes: bool
    bounding_box: Any
    file: str


class SplitResponse(BaseModel):
    status: str
    num_parts: int
    fits_in_volume: bool
    split_axes: Any
    parts: List[SplitPartInfo]


class FormatsResponse(BaseModel):
    formats: Dict[str, Any]
    printers: Dict[str, str]


class PrinterListResponse(BaseModel):
    printers: List[Any]


class PrinterSendResponse(BaseModel):
    status: str
    message: str


class CacheStatsResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    hits: int = Field(..., description="Number of cache hits since startup")
    misses: int = Field(..., description="Number of cache misses since startup")
    size_bytes: int = Field(..., description="Total size of cached files in bytes")
    num_entries: int = Field(..., description="Number of cached mesh entries")


# ── OpenAPI tag metadata ──────────────────────────────────────────

tags_metadata = [
    {
        "name": "Generation",
        "description": "Endpoints for generating 3D models from images or text descriptions.",
    },
    {
        "name": "Optimization",
        "description": "Endpoints for analyzing, optimizing, costing, and splitting 3D meshes for printing.",
    },
    {
        "name": "Printers",
        "description": "Endpoints for discovering and sending jobs to Bambu Lab printers.",
    },
    {
        "name": "System",
        "description": "Health checks, supported formats, and cache management.",
    },
]


app = FastAPI(
    title="PrintForge",
    description="One photo to 3D print — commercial API. Upload an image and receive a watertight, "
                "print-ready 3D model in 3MF/STL/OBJ format.",
    version="1.1.0",
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (loaded once)
_pipeline: Optional[PrintForgePipeline] = None
_cache: Optional[ImageCache] = None


def get_pipeline() -> PrintForgePipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = PrintForgePipeline(PipelineConfig())
    return _pipeline


def get_cache() -> ImageCache:
    global _cache
    if _cache is None:
        _cache = ImageCache()
    return _cache


# ── Generation endpoints ──────────────────────────────────────────

@app.post(
    "/api/generate",
    tags=["Generation"],
    summary="Generate a 3D printable file from an image",
    description="Upload a photo (JPEG/PNG) and receive a watertight 3D mesh file "
                "suitable for FDM/SLA printing. Supports 3MF, STL, and OBJ output.",
    response_class=FileResponse,
    responses={
        200: {
            "description": "The generated 3D mesh file",
            "headers": {
                "X-PrintForge-Vertices": {"description": "Number of vertices", "schema": {"type": "integer"}},
                "X-PrintForge-Faces": {"description": "Number of faces", "schema": {"type": "integer"}},
                "X-PrintForge-Watertight": {"description": "Whether mesh is watertight", "schema": {"type": "string"}},
                "X-PrintForge-Duration-Ms": {"description": "Pipeline duration in ms", "schema": {"type": "integer"}},
            },
        },
        400: {"description": "Invalid input (not an image)"},
        500: {"description": "Pipeline failure"},
    },
)
async def generate(
    image: UploadFile = File(...),
    format: str = "3mf",
    size_mm: float = 50.0,
    add_base: bool = False,
):
    """Upload an image, get a 3D printable file."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image (jpg/png)")

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        content = await image.read()
        tmp_in.write(content)
        input_path = tmp_in.name

    output_suffix = ".3mf" if format == "3mf" else ".stl"
    with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as tmp_out:
        output_path = tmp_out.name

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


@app.post(
    "/api/text-to-3d",
    tags=["Generation"],
    summary="Generate a 3D model from a text description",
    description="Provide a text description of the desired object and optionally an image. "
                "Returns a 3D mesh file, or a fallback prompt if image generation is unavailable.",
    response_class=FileResponse,
    responses={
        200: {
            "description": "The generated 3D mesh file, or a JSON fallback response",
            "content": {
                "application/octet-stream": {},
                "application/json": {
                    "example": {
                        "status": "fallback",
                        "prompt": "A small dragon figurine...",
                        "message": "Image generation unavailable. Use the prompt to generate an image manually.",
                    }
                },
            },
        },
        500: {"description": "Text-to-3D pipeline failure"},
    },
)
async def text_to_3d(
    description: str = Form(...),
    format: str = Form("3mf"),
    size_mm: float = Form(50.0),
    image: Optional[UploadFile] = File(None),
):
    """Generate a 3D model from a text description.

    Optionally provide an image to skip the image generation step.
    """
    from .text_to_3d import TextTo3DPipeline, TextTo3DConfig

    output_suffix = f".{format}" if format in ("3mf", "stl", "obj") else ".3mf"
    with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as tmp_out:
        output_path = tmp_out.name

    image_path = None
    if image and image.filename:
        suffix = Path(image.filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_img:
            content = await image.read()
            tmp_img.write(content)
            image_path = tmp_img.name

    try:
        pipeline_config = PipelineConfig(
            output_format=format,
            scale_mm=size_mm,
        )
        text_pipeline = TextTo3DPipeline()
        result = text_pipeline.run(
            description=description,
            output_path=output_path,
            image_path=image_path,
            pipeline_config=pipeline_config,
        )

        if result.used_fallback:
            return JSONResponse({
                "status": "fallback",
                "prompt": result.prompt_used,
                "message": "Image generation unavailable. Use the prompt to generate an image manually.",
            })

        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"printforge_text3d{output_suffix}",
            headers={
                "X-PrintForge-Prompt": result.prompt_used[:200],
            },
        )
    except Exception as e:
        logger.exception("Text-to-3D failed")
        raise HTTPException(500, f"Text-to-3D failed: {str(e)}")


@app.post(
    "/api/analyze",
    tags=["Generation"],
    summary="Analyze an image without generating a mesh",
    description="Upload an image and receive metadata and analysis without running full 3D inference.",
    response_model=AnalyzeResponse,
    responses={
        400: {"description": "Invalid input (not an image)"},
    },
)
async def analyze(image: UploadFile = File(...)):
    """Analyze an image and return metadata without generating the full mesh."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    return JSONResponse({
        "status": "ok",
        "message": "Analysis endpoint — coming soon",
        "supported_formats": ["3mf", "stl", "obj"],
        "default_size_mm": 50.0,
    })


# ── Optimization endpoints ────────────────────────────────────────

@app.post(
    "/api/optimize",
    tags=["Optimization"],
    summary="Optimize a mesh for 3D printing",
    description="Upload a mesh file and get orientation analysis, material estimates, "
                "and printability issue detection for a specific printer.",
    response_model=OptimizeResponse,
    responses={
        500: {"description": "Optimization failure"},
    },
)
async def optimize(
    mesh_file: UploadFile = File(...),
    printer: str = Form("bambu-a1"),
    infill: float = Form(0.15),
    layer_height: float = Form(0.2),
    material: str = Form("pla"),
):
    """Upload a mesh file and get print analysis and optimization suggestions."""
    import trimesh

    suffix = Path(mesh_file.filename or "model.stl").suffix or ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await mesh_file.read()
        tmp.write(content)
        mesh_path = tmp.name

    try:
        mesh = trimesh.load(mesh_path)

        preset = PRINTER_PRESETS.get(printer, PRINTER_PRESETS["bambu-a1"])
        build_vol = preset["volume"]

        optimizer = PrintOptimizer()

        orientation = optimizer.find_best_orientation(mesh)
        estimate = optimizer.estimate_material(
            mesh, infill=infill, layer_height=layer_height, material=material,
        )
        issues = optimizer.check_printability(mesh, build_volume=build_vol)

        return JSONResponse({
            "status": "ok",
            "printer": preset["name"],
            "orientation": {
                "height_mm": round(orientation.height, 1),
                "support_estimate_mm3": round(orientation.support_volume_estimate, 0),
                "base_area_mm2": round(orientation.base_area, 1),
                "score": round(orientation.score, 2),
                "overhang_percentage": round(orientation.overhang_percentage, 1),
                "support_contact_area_mm2": round(orientation.support_contact_area, 1),
            },
            "estimate": {
                "print_time_minutes": estimate.print_time_minutes,
                "filament_grams": estimate.filament_grams,
                "filament_meters": estimate.filament_meters,
                "layer_count": estimate.layer_count,
            },
            "issues": [
                {"severity": i.severity, "category": i.category, "message": i.message}
                for i in issues
            ],
            "mesh_info": {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "is_watertight": bool(mesh.is_watertight),
                "bounding_box_mm": mesh.bounding_box.extents.tolist(),
            },
        })
    except Exception as e:
        logger.exception("Optimization failed")
        raise HTTPException(500, f"Optimization failed: {str(e)}")


@app.post(
    "/api/cost",
    tags=["Optimization"],
    summary="Estimate printing cost for a mesh",
    description="Upload a mesh and get a detailed cost breakdown including filament usage, "
                "print time, and electricity cost.",
    response_model=CostResponse,
    responses={
        500: {"description": "Cost estimation failure"},
    },
)
async def cost_estimate(
    mesh_file: UploadFile = File(...),
    material: str = Form("PLA"),
    infill: float = Form(0.15),
    layer_height: float = Form(0.2),
):
    """Upload a mesh and get a full cost estimate."""
    import trimesh
    from .cost_estimator import CostEstimator

    suffix = Path(mesh_file.filename or "model.stl").suffix or ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await mesh_file.read()
        tmp.write(content)
        mesh_path = tmp.name

    try:
        mesh = trimesh.load(mesh_path)
        estimator = CostEstimator()
        est = estimator.estimate(mesh, material=material, infill=infill, layer_height=layer_height)

        return JSONResponse({
            "status": "ok",
            "material": material.upper(),
            "infill_percent": round(infill * 100, 1),
            "layer_height_mm": layer_height,
            "cost": {
                "filament_grams": est.filament_grams,
                "filament_meters": est.filament_meters,
                "filament_cost_usd": est.filament_cost_usd,
                "print_time_hours": est.print_time_hours,
                "electricity_cost_usd": est.electricity_cost_usd,
                "total_cost_usd": est.total_cost_usd,
            },
            "mesh_info": {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "is_watertight": bool(mesh.is_watertight),
                "bounding_box_mm": mesh.bounding_box.extents.tolist(),
            },
        })
    except Exception as e:
        logger.exception("Cost estimation failed")
        raise HTTPException(500, f"Cost estimation failed: {str(e)}")


@app.post(
    "/api/split",
    tags=["Optimization"],
    summary="Split a mesh into printer-sized parts",
    description="Upload a mesh that exceeds your build volume and split it into interlocking parts "
                "with alignment pins and holes.",
    response_model=SplitResponse,
    responses={
        500: {"description": "Split operation failure"},
    },
)
async def split(
    mesh_file: UploadFile = File(...),
    volume: str = Form("256x256x256"),
):
    """Upload a mesh and split it into parts that fit the build volume."""
    import trimesh

    from .part_splitter import PartSplitter, SplitConfig, BuildVolume

    suffix = Path(mesh_file.filename or "model.stl").suffix or ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await mesh_file.read()
        tmp.write(content)
        mesh_path = tmp.name

    try:
        mesh = trimesh.load(mesh_path)
        build_vol = BuildVolume.from_string(volume)

        splitter = PartSplitter(SplitConfig(build_volume=build_vol))
        result = splitter.split(mesh)

        # Save split parts as separate files
        part_paths = []
        for part in result.parts:
            with tempfile.NamedTemporaryFile(
                suffix=".stl", delete=False, prefix=f"part{part.part_index}_"
            ) as tmp_part:
                part.mesh.export(tmp_part.name, file_type="stl")
                part_paths.append(tmp_part.name)

        return JSONResponse({
            "status": "ok",
            "num_parts": result.num_parts,
            "fits_in_volume": result.fits_in_volume,
            "split_axes": result.split_axes,
            "parts": [
                {
                    "index": p.part_index,
                    "has_pins": p.has_pins,
                    "has_holes": p.has_holes,
                    "bounding_box": p.bounding_box,
                    "file": part_paths[i],
                }
                for i, p in enumerate(result.parts)
            ],
        })
    except Exception as e:
        logger.exception("Split failed")
        raise HTTPException(500, f"Split failed: {str(e)}")


# ── Printer endpoints ─────────────────────────────────────────────

@app.get(
    "/api/printers",
    tags=["Printers"],
    summary="Discover printers on the local network",
    description="Scan the local network for Bambu Lab printers via SSDP discovery.",
    response_model=PrinterListResponse,
)
async def discover_printers():
    """Discover Bambu Lab printers on the local network via SSDP."""
    from .bambu import BambuConnection, BambuPrinter

    conn = BambuConnection(BambuPrinter(ip="", access_code=""))
    found = conn.discover(timeout=3.0)
    return JSONResponse({"printers": found})


@app.post(
    "/api/printers/send",
    tags=["Printers"],
    summary="Send a print job to a Bambu printer",
    description="Upload a 3MF or gcode file and send it directly to a Bambu Lab printer by IP address.",
    response_model=PrinterSendResponse,
    responses={
        502: {"description": "Printer unreachable"},
        500: {"description": "Failed to send print job"},
    },
)
async def send_to_printer(
    file: UploadFile = File(...),
    printer_ip: str = Form(...),
    access_code: str = Form(...),
):
    """Send a 3MF/gcode file to a Bambu printer by IP."""
    from .bambu import BambuConnection, BambuPrinter, PrintJob

    suffix = Path(file.filename or "model.3mf").suffix or ".3mf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        file_path = tmp.name

    printer = BambuPrinter(ip=printer_ip, access_code=access_code)
    conn = BambuConnection(printer)

    connected = conn.connect()
    if not connected:
        raise HTTPException(502, f"Cannot reach printer at {printer_ip}")

    job = PrintJob(file_path=file_path)
    success = conn.send_print(job)
    if not success:
        raise HTTPException(500, "Failed to send print job")

    return JSONResponse({"status": "ok", "message": f"Print job sent to {printer_ip}"})


# ── System endpoints ──────────────────────────────────────────────

@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
    description="Returns service status and version. Use for liveness probes.",
    response_model=HealthResponse,
)
async def health():
    return {"status": "ok", "version": "1.1.0"}


@app.get(
    "/api/formats",
    tags=["System"],
    summary="List supported export formats and printers",
    description="Returns all supported 3D file formats and known printer presets.",
    response_model=FormatsResponse,
)
async def list_formats():
    """List supported export formats."""
    return JSONResponse({
        "formats": SUPPORTED_FORMATS,
        "printers": {k: v["name"] for k, v in PRINTER_PRESETS.items()},
    })


@app.get(
    "/api/cache/stats",
    tags=["System"],
    summary="Get image cache statistics",
    description="Returns cache hit/miss counts, total size, and number of cached entries.",
    response_model=CacheStatsResponse,
)
async def cache_stats():
    """Return current cache statistics."""
    cache = get_cache()
    s = cache.stats()
    return JSONResponse({
        "status": "ok",
        "hits": s.hits,
        "misses": s.misses,
        "size_bytes": s.size_bytes,
        "num_entries": s.num_entries,
    })


# ── WebSocket progress endpoint ────────────────────────────────────

@app.websocket("/ws/progress")
async def ws_progress(websocket: WebSocket):
    """WebSocket endpoint for real-time pipeline progress.

    Client sends a JSON message to start generation:
        {"action": "generate", "format": "3mf", "size_mm": 50, "add_base": false}
    Then streams progress updates:
        {"stage": "inference", "progress": 0.25}
    And a final completion message:
        {"stage": "done", "progress": 1.0, "result": {...}}
    """
    await websocket.accept()

    try:
        msg = await websocket.receive_json()
        if msg.get("action") != "generate":
            await websocket.send_json({"error": "Unknown action"})
            await websocket.close()
            return

        # Expect the image was uploaded via /api/generate first,
        # or a path is provided. For simplicity, stream progress
        # for a pipeline run referenced by temp path.
        image_path = msg.get("image_path")
        if not image_path:
            await websocket.send_json({"error": "image_path required"})
            await websocket.close()
            return

        fmt = msg.get("format", "3mf")
        output_suffix = ".3mf" if fmt == "3mf" else ".stl"
        with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as tmp_out:
            output_path = tmp_out.name

        pipeline = get_pipeline()
        pipeline.config.output_format = fmt
        pipeline.config.scale_mm = msg.get("size_mm", 50.0)
        pipeline.config.add_base = msg.get("add_base", False)

        loop = asyncio.get_event_loop()

        async def send_progress(stage: str, progress: float):
            await websocket.send_json({"stage": stage, "progress": progress})

        def progress_callback(stage: str, progress: float):
            asyncio.run_coroutine_threadsafe(send_progress(stage, progress), loop)

        result = await loop.run_in_executor(
            None, lambda: pipeline.run(image_path, output_path, progress_callback=progress_callback)
        )

        await websocket.send_json({
            "stage": "done",
            "progress": 1.0,
            "result": {
                "mesh_path": result.mesh_path,
                "vertices": result.vertices,
                "faces": result.faces,
                "is_watertight": result.is_watertight,
                "duration_ms": result.duration_ms,
            },
        })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# ── Static files & startup ─────────────────────────────────────────

import os
_web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web")

if os.path.isdir(_web_dir):
    from fastapi.responses import HTMLResponse

    @app.get("/", response_class=HTMLResponse)
    async def web_ui():
        index_path = os.path.join(_web_dir, "index.html")
        if os.path.exists(index_path):
            with open(index_path) as f:
                return f.read()
        return "<h1>PrintForge</h1><p>Web UI not found.</p>"

    # Serve all static assets (JS, CSS, images) from web/
    app.mount("/static", StaticFiles(directory=_web_dir), name="static")


def start():
    """Start the web server."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
