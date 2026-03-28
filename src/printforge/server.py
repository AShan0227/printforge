"""PrintForge Web API — Upload image, get 3D print file."""

import asyncio
import json
import tempfile
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from starlette.requests import Request
from starlette.responses import Response

from .pipeline import PrintForgePipeline, PipelineConfig
from .print_optimizer import PrintOptimizer, PRINTER_PRESETS
from .formats import SUPPORTED_FORMATS
from .cache import ImageCache
from .safety import RateLimiter
from .api_v2 import register_user as api_register_user, create_api_key, validate_api_key, increment_usage, get_key_stats, decode_jwt, _make_jwt, login_user
from .billing import record_usage, get_usage_history, get_monthly_usage
from .feishu_notifier import send_notification, GenerationResult
from .metrics import get_metrics
from .health import get_health_checker
from .model_store import store_model, list_models, get_model, delete_model, get_user_stats
from .queue import GenerationQueue
from .rate_limit import generate_limiter, ip_limiter
from .webhook import register_webhook, unregister_webhook, list_webhooks, fire_event, WebhookEvent
from .sse import event_bus, emit_generation_started, emit_generation_done, emit_generation_failed

logger = logging.getLogger(__name__)

# Global rate limiter for generation endpoints
_rate_limiter = RateLimiter(max_requests=20, window_seconds=3600)


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


class QualityResponse(BaseModel):
    status: str
    total_score: float
    grade: str
    watertight: Dict[str, Any]
    face_count: Dict[str, Any]
    aspect_ratio: Dict[str, Any]
    thin_walls: Dict[str, Any]
    overhangs: Dict[str, Any]


class RepairResponse(BaseModel):
    status: str
    was_watertight: bool
    is_watertight: bool
    faces_before: int
    faces_after: int
    repairs: List[str]
    used_voxel_remesh: bool


class BatchItemResponse(BaseModel):
    filename: str
    success: bool
    error: Optional[str] = None


class BatchResponse(BaseModel):
    status: str
    total: int
    succeeded: int
    failed: int
    duration_ms: float
    items: List[BatchItemResponse]


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


class TermsResponse(BaseModel):
    text: str


class BenchmarkReportResponse(BaseModel):
    status: str
    report: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class InfoResponse(BaseModel):
    version: str = Field(..., json_schema_extra={"example": "1.4.0"})
    description: str = Field(..., json_schema_extra={"example": "One photo to 3D print"})
    features: List[str] = Field(..., json_schema_extra={"example": ["image-to-3d", "text-to-3d"]})
    supported_formats: List[str] = Field(..., json_schema_extra={"example": ["3mf", "stl", "obj"]})
    supported_printers: Dict[str, str] = Field(default_factory=dict)


class FailureRiskResponse(BaseModel):
    type: str = Field(..., json_schema_extra={"example": "thin_wall"})
    severity: str = Field(..., json_schema_extra={"example": "high"})
    location: str = Field(..., json_schema_extra={"example": "Minimum thickness: 0.3mm"})
    suggestion: str = Field(..., json_schema_extra={"example": "Increase wall thickness to 0.4mm"})


class PredictResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    risk_score: float = Field(..., json_schema_extra={"example": 45.0})
    risk_level: str = Field(..., json_schema_extra={"example": "medium"})
    risks: List[FailureRiskResponse] = Field(default_factory=list)


class StatsResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    total_events: int = Field(..., json_schema_extra={"example": 150})
    generations: int = Field(..., json_schema_extra={"example": 42})
    formats: Dict[str, int] = Field(default_factory=dict)
    backends: Dict[str, int] = Field(default_factory=dict)
    avg_duration_ms: Optional[float] = None
    avg_quality_score: Optional[float] = None


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
    {
        "name": "Legal",
        "description": "Terms of service and acceptance tracking.",
    },
    {
        "name": "Benchmark",
        "description": "Performance benchmarking endpoints.",
    },
    {
        "name": "Safety",
        "description": "Content safety, rate limiting, and failure prediction.",
    },
    {
        "name": "Analytics",
        "description": "Local usage analytics and telemetry.",
    },
]


app = FastAPI(
    title="PrintForge",
    description="One photo to 3D print — commercial API. Upload an image and receive a watertight, "
                "print-ready 3D model in 3MF/STL/OBJ format.",
    version="2.2.0",
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-PrintForge-Vertices", "X-PrintForge-Faces", "X-PrintForge-Watertight",
        "X-PrintForge-Duration-Ms", "X-PrintForge-Multi-View",
        "X-PrintForge-Stages", "X-PrintForge-Warnings",
    ],
)

# v2.2 middleware
from .middleware import RequestLoggingMiddleware, RateLimitMiddleware, ErrorHandlerMiddleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(ErrorHandlerMiddleware)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limit generation endpoints to 20 requests/hour per IP."""
    rate_limited_paths = {"/api/generate", "/api/text-to-3d", "/api/batch"}
    if request.url.path in rate_limited_paths and request.method == "POST":
        client_ip = request.client.host if request.client else "unknown"
        allowed, remaining = _rate_limiter.check(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded. Max 20 generations per hour."},
                headers={"Retry-After": "3600", "X-RateLimit-Remaining": "0"},
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Limit"] = "20"
        return response
    return await call_next(request)

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
    request: Request,
    image: UploadFile = File(...),
    format: str = "3mf",
    size_mm: float = 50.0,
    add_base: bool = False,
    backend: str = "auto",
    multi_view: bool = False,
    use_depth: bool = True,
):
    """Upload an image, get a 3D printable file.

    Requires API key via X-API-Key header (optional for now, enforced when configured).
    """
    # API Key auth (optional — if key provided, validate and track usage)
    api_key_raw = request.headers.get("X-API-Key")
    api_key_obj = None
    if api_key_raw:
        api_key_obj = validate_api_key(api_key_raw)
        if api_key_obj is None:
            raise HTTPException(401, "Invalid API key")
        if api_key_obj.quota_exhausted:
            raise HTTPException(429, "API key quota exhausted. Upgrade your plan or contact support.")

    # Validate backend parameter
    valid_backends = ("auto", "hunyuan3d", "api", "local", "placeholder")
    if backend not in valid_backends:
        raise HTTPException(400, f"Invalid backend: {backend}. Must be one of: {', '.join(valid_backends)}")

    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        content = await image.read()

        # Validate magic bytes and file size
        from .safety import validate_image_magic_bytes, VALID_IMAGE_FORMATS
        is_valid, detected = validate_image_magic_bytes(content)
        if not is_valid:
            accepted = ", ".join(sorted(VALID_IMAGE_FORMATS))
            raise HTTPException(
                400,
                f"File is not a valid image (detected: {detected}). "
                f"Accepted formats: {accepted}",
            )
        if len(content) > 50 * 1024 * 1024:
            raise HTTPException(400, "File too large (max 50MB)")

        tmp_in.write(content)
        input_path = tmp_in.name

    format_map = {"3mf": ".3mf", "stl": ".stl", "obj": ".obj", "glb": ".glb"}
    output_suffix = format_map.get(format, ".stl")
    with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        config = PipelineConfig(
            inference_backend=backend,
            output_format=format,
            scale_mm=size_mm,
            add_base=add_base,
            multi_view=multi_view,
            use_depth=use_depth,
        )
        pipeline = PrintForgePipeline(config)
        result = pipeline.run(input_path, output_path)

        # Record billing and store model
        user_id = api_key_obj.user_id if api_key_obj else "anonymous"
        if api_key_raw and api_key_obj:
            try:
                record_usage(api_key_raw, "generate_3d", True, int(result.duration_ms), backend)
            except Exception as billing_err:
                logger.warning(f"Billing record failed: {billing_err}")

        try:
            stored = store_model(
                user_id=user_id,
                input_filename=image.filename or "upload.png",
                output_path=output_path,
                output_format=format,
                vertices=result.vertices,
                faces=result.faces,
                is_watertight=result.is_watertight,
                duration_ms=int(result.duration_ms),
                backend=backend,
                multi_view=multi_view,
            )
            logger.info(f"Model stored: {stored.model_id}")
        except Exception as store_err:
            logger.warning(f"Model store failed: {store_err}")

        # Send Feishu notification (async, best-effort)
        try:
            send_notification(GenerationResult(
                success=True,
                model_file=output_path,
                preview_url=f"/preview?url=/static/{Path(output_path).name}",
                operation="generate_3d",
                duration_ms=int(result.duration_ms),
                user_email=api_key_obj.user_id if api_key_obj else None,
            ))
        except Exception:
            pass  # non-critical

        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"printforge_output{output_suffix}",
            headers={
                "X-PrintForge-Vertices": str(result.vertices),
                "X-PrintForge-Faces": str(result.faces),
                "X-PrintForge-Watertight": str(result.is_watertight),
                "X-PrintForge-Duration-Ms": str(int(result.duration_ms)),
                "X-PrintForge-Multi-View": str(multi_view),
                "X-PrintForge-Stages": json.dumps(result.stages),
                "X-PrintForge-Warnings": json.dumps(result.warnings),
            },
        )
    except RuntimeError as e:
        err_msg = str(e).lower()
        if "quota" in err_msg or "rate" in err_msg:
            raise HTTPException(
                503,
                "Hunyuan3D quota exceeded. Try again later or set the HF_TOKEN "
                "environment variable for higher limits.",
            )
        if "hunyuan" in err_msg or "gradio" in err_msg or "space" in err_msg:
            raise HTTPException(
                503,
                "Hunyuan3D Space is currently unavailable. Try again later "
                "or set HF_TOKEN env var for authenticated access.",
            )
        # Identify which pipeline stage failed when possible
        raise HTTPException(500, f"Pipeline failed at runtime: {e}")
    except Exception as e:
        logger.exception("Pipeline failed")
        stage_hint = ""
        err_str = str(e).lower()
        if "load" in err_str or "image" in err_str:
            stage_hint = " (stage: image loading)"
        elif "infer" in err_str or "model" in err_str:
            stage_hint = " (stage: 3D inference)"
        elif "watertight" in err_str or "voxel" in err_str:
            stage_hint = " (stage: watertight conversion)"
        elif "export" in err_str:
            stage_hint = " (stage: mesh export)"
        raise HTTPException(500, f"Generation failed{stage_hint}: {e}")


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


# ── Quality endpoint ──────────────────────────────────────────────

@app.post(
    "/api/quality",
    tags=["Optimization"],
    summary="Score mesh quality for 3D printing",
    description="Upload a mesh and get a quality score (0-100) with per-metric breakdown.",
    response_model=QualityResponse,
    responses={500: {"description": "Quality scoring failure"}},
)
async def quality_score(mesh_file: UploadFile = File(...)):
    """Upload a mesh and get a quality score."""
    import trimesh
    from .quality import QualityScorer

    suffix = Path(mesh_file.filename or "model.stl").suffix or ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await mesh_file.read()
        tmp.write(content)
        mesh_path = tmp.name

    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        scorer = QualityScorer()
        report = scorer.score(mesh)

        return JSONResponse({
            "status": "ok",
            "total_score": report.total_score,
            "grade": report.grade,
            "watertight": {"score": report.watertight_score, "max": 30, "value": report.is_watertight},
            "face_count": {"score": report.face_count_score, "max": 20, "value": report.face_count},
            "aspect_ratio": {"score": report.aspect_ratio_score, "max": 15, "value": report.aspect_ratio},
            "thin_walls": {"score": report.thin_wall_score, "max": 20, "min_mm": report.min_thickness_mm},
            "overhangs": {"score": report.overhang_score, "max": 15, "percentage": report.overhang_percentage},
        })
    except Exception as e:
        logger.exception("Quality scoring failed")
        raise HTTPException(500, f"Quality scoring failed: {str(e)}")


# ── Repair endpoint ──────────────────────────────────────────────

@app.post(
    "/api/repair",
    tags=["Optimization"],
    summary="Repair a broken mesh for 3D printing",
    description="Upload a broken mesh and receive a repaired, watertight version.",
    response_class=FileResponse,
    responses={
        200: {"description": "The repaired mesh file"},
        500: {"description": "Repair failure"},
    },
)
async def repair_mesh(mesh_file: UploadFile = File(...)):
    """Upload a broken mesh and get a repaired version."""
    import trimesh
    from .repair import MeshRepair

    suffix = Path(mesh_file.filename or "model.stl").suffix or ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await mesh_file.read()
        tmp.write(content)
        mesh_path = tmp.name

    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        repairer = MeshRepair()
        repaired, report = repairer.repair(mesh)

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp_out:
            repaired.export(tmp_out.name, file_type="stl")
            output_path = tmp_out.name

        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename="repaired.stl",
            headers={
                "X-PrintForge-Watertight-Before": str(report.was_watertight_before),
                "X-PrintForge-Watertight-After": str(report.is_watertight_after),
                "X-PrintForge-Repairs": str(len(report.repairs_performed)),
                "X-PrintForge-Voxel-Remesh": str(report.used_voxel_remesh),
            },
        )
    except Exception as e:
        logger.exception("Repair failed")
        raise HTTPException(500, f"Repair failed: {str(e)}")


# ── Batch endpoint ───────────────────────────────────────────────

@app.post(
    "/api/batch",
    tags=["Generation"],
    summary="Batch process multiple images to 3D models",
    description="Upload multiple images and receive per-item processing results.",
    response_model=BatchResponse,
    responses={500: {"description": "Batch processing failure"}},
)
async def batch_process(
    images: List[UploadFile] = File(...),
    format: str = Form("3mf"),
    size_mm: float = Form(50.0),
):
    """Upload multiple images for batch 3D generation."""
    from .batch import BatchProcessor

    with tempfile.TemporaryDirectory() as tmp_input, \
         tempfile.TemporaryDirectory() as tmp_output:

        image_paths = []
        for img in images:
            if not img.content_type or not img.content_type.startswith("image/"):
                continue
            suffix = Path(img.filename or "upload.jpg").suffix or ".jpg"
            path = Path(tmp_input) / f"{len(image_paths):04d}{suffix}"
            content = await img.read()
            path.write_bytes(content)
            image_paths.append(str(path))

        if not image_paths:
            raise HTTPException(400, "No valid images provided")

        try:
            config = PipelineConfig(scale_mm=size_mm, output_format=format)
            processor = BatchProcessor(config=config, max_workers=3)
            result = processor.process(image_paths, tmp_output, format)

            return JSONResponse({
                "status": "ok",
                "total": len(result.items),
                "succeeded": result.succeeded,
                "failed": result.failed,
                "duration_ms": round(result.total_duration_ms, 0),
                "items": [
                    {
                        "filename": Path(item.input_path).name,
                        "success": item.success,
                        "error": item.error,
                    }
                    for item in result.items
                ],
            })
        except Exception as e:
            logger.exception("Batch processing failed")
            raise HTTPException(500, f"Batch processing failed: {str(e)}")


# ── Printer endpoints ─────────────────────────────────────────────

@app.get(
    "/api/printers/profiles",
    tags=["Printers"],
    summary="List all supported printer profiles",
    description="Returns detailed specs for all supported printer models.",
)
async def printer_profiles():
    """List all supported printer profiles with specs."""
    from .printer_profiles import PRINTER_DB
    return JSONResponse({
        "profiles": {
            key: {
                "name": p.name,
                "build_volume": {"x": p.build_volume[0], "y": p.build_volume[1], "z": p.build_volume[2]},
                "max_speed": p.max_speed,
                "heated_bed": p.heated_bed,
                "auto_level": p.auto_level,
                "nozzle_sizes": p.nozzle_sizes,
                "default_layer_height": p.default_layer_height,
                "default_infill": p.default_infill,
            }
            for key, p in sorted(PRINTER_DB.items())
        }
    })


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
    return {"status": "ok", "version": "1.4.0"}


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


# ── Terms of Service endpoints ─────────────────────────────────────

@app.get(
    "/api/terms",
    tags=["Legal"],
    summary="Get Terms of Service",
    description="Returns the full Terms of Service text.",
    response_model=TermsResponse,
)
async def get_terms():
    """Return the Terms of Service."""
    from .legal import get_tos
    return JSONResponse({"text": get_tos()})


@app.get(
    "/api/terms/accept",
    tags=["Legal"],
    summary="Confirm TOS acceptance",
    description="Returns a confirmation that TOS acceptance was recorded. "
                "Actual tracking is done client-side via localStorage.",
)
async def accept_terms():
    """Acknowledge TOS acceptance (tracking is client-side via localStorage)."""
    return JSONResponse({"status": "accepted", "message": "Terms accepted. Tracked in browser localStorage."})


# ── Benchmark endpoints ───────────────────────────────────────────

@app.get(
    "/api/benchmark/latest",
    tags=["Benchmark"],
    summary="Get latest benchmark report",
    description="Returns the most recently saved benchmark report, if any.",
    response_model=BenchmarkReportResponse,
)
async def benchmark_latest():
    """Return the latest benchmark report."""
    from .benchmark import BenchmarkSuite
    report = BenchmarkSuite.load_latest()
    if report is None:
        return JSONResponse({"status": "none", "message": "No benchmark report found. Run: printforge benchmark <image>"})
    return JSONResponse({"status": "ok", "report": report.to_dict()})


# ── Info endpoint ─────────────────────────────────────────────────

@app.get(
    "/api/info",
    tags=["System"],
    summary="Get API info, version, and feature list",
    description="Returns version, supported features, formats, and printers.",
    response_model=InfoResponse,
)
async def api_info():
    """Return API version, features, and capabilities."""
    return JSONResponse({
        "version": "1.4.0",
        "description": "One photo to 3D print — commercial API",
        "features": [
            "image-to-3d",
            "text-to-3d",
            "multi-view-enhance",
            "batch-processing",
            "print-optimization",
            "cost-estimation",
            "part-splitting",
            "quality-scoring",
            "mesh-repair",
            "failure-prediction",
            "analytics",
            "competitor-monitoring",
            "content-safety",
            "rate-limiting",
            "bambu-lab-integration",
            "websocket-progress",
            "video-to-3d-frames",
        ],
        "supported_formats": ["3mf", "stl", "obj"],
        "supported_printers": {k: v["name"] for k, v in PRINTER_PRESETS.items()},
    })


# ── Prediction endpoint ──────────────────────────────────────────

@app.post(
    "/api/predict",
    tags=["Safety"],
    summary="Predict print failures for a mesh",
    description="Upload a mesh and get a failure risk score (0-100) with specific risks identified.",
    response_model=PredictResponse,
    responses={500: {"description": "Prediction failure"}},
)
async def predict_failure(
    mesh_file: UploadFile = File(...),
    material: str = Form("PLA"),
    layer_height: float = Form(0.2),
    printer: str = Form("a1"),
):
    """Upload a mesh and predict print failures."""
    import trimesh
    from .failure_predictor import FailurePredictor

    suffix = Path(mesh_file.filename or "model.stl").suffix or ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await mesh_file.read()
        tmp.write(content)
        mesh_path = tmp.name

    try:
        mesh = trimesh.load(mesh_path, force="mesh")
        predictor = FailurePredictor()
        prediction = predictor.predict(
            mesh, material=material, layer_height=layer_height, printer=printer,
        )

        return JSONResponse({
            "status": "ok",
            "risk_score": prediction.risk_score,
            "risk_level": prediction.risk_level,
            "risks": [
                {
                    "type": r.type,
                    "severity": r.severity,
                    "location": r.location,
                    "suggestion": r.suggestion,
                }
                for r in prediction.risks
            ],
        })
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(500, f"Prediction failed: {str(e)}")


# ── Analytics endpoint ────────────────────────────────────────────

@app.get(
    "/api/stats",
    tags=["Analytics"],
    summary="Get usage analytics summary",
    description="Returns local analytics: generation counts, format/backend breakdowns, average durations.",
    response_model=StatsResponse,
)
async def get_stats():
    """Return local usage analytics."""
    from .analytics import Analytics
    analytics = Analytics()
    stats = analytics.get_stats()
    return JSONResponse({
        "status": "ok",
        **stats,
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
            "download_url": f"/static/{Path(output_path).name}",
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


# ── API v2: Auth & Billing ──────────────────────────────────────────

@app.post("/api/v2/register", tags=["Auth"])
async def register_endpoint(request: Request):
    """Register a new user and get API key."""
    body = await request.json()
    username = body.get("username")
    password = body.get("password")
    email = body.get("email", "")
    if not username or not password:
        raise HTTPException(400, "username and password required")
    
    user, raw_key = api_register_user(username, password, email)
    token = _make_jwt(user.user_id, username)
    
    return JSONResponse({
        "status": "ok",
        "user_id": user.user_id,
        "api_key": raw_key,
        "jwt_token": token,
        "message": "Save your API key — it won't be shown again.",
    })


@app.post("/api/v2/login", tags=["Auth"])
async def login_endpoint(request: Request):
    """Login and get JWT + API key."""
    body = await request.json()
    username = body.get("username")
    password = body.get("password")
    if not username or not password:
        raise HTTPException(400, "username and password required")
    
    result = login_user(username, password)
    if not result:
        raise HTTPException(401, "Invalid credentials")
    
    user, token, api_key = result
    return JSONResponse({
        "user_id": user.user_id,
        "jwt_token": token,
        "api_key": api_key,
    })


@app.get("/api/v2/quota", tags=["Auth"])
async def check_quota(request: Request):
    """Check remaining quota for an API key."""
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not api_key:
        raise HTTPException(401, "X-API-Key header or api_key param required")
    stats = get_key_stats(api_key)
    if not stats:
        raise HTTPException(401, "Invalid API key")
    return JSONResponse(stats)


@app.get("/api/v2/usage", tags=["Billing"])
async def usage_history_endpoint(request: Request, limit: int = 50):
    """Get usage history for an API key."""
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not api_key:
        raise HTTPException(401, "X-API-Key header required")
    return JSONResponse({"usage": get_usage_history(api_key, limit)})


@app.get("/api/v2/usage/monthly", tags=["Billing"])
async def monthly_usage_endpoint(request: Request):
    """Get this month's usage stats."""
    api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
    if not api_key:
        raise HTTPException(401, "X-API-Key header required")
    return JSONResponse(get_monthly_usage(api_key))


@app.post("/api/v2/keys", tags=["Auth"])
async def create_new_key(request: Request):
    """Create an additional API key (requires JWT)."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Bearer token required")
    
    payload = decode_jwt(auth[7:])
    if not payload:
        raise HTTPException(401, "Invalid or expired token")
    
    raw_key = create_api_key(payload["sub"])
    return JSONResponse({
        "api_key": raw_key,
        "message": "Save your API key — it won't be shown again.",
    })


# ── Static files & startup ─────────────────────────────────────────

import os
_web_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web")


# ── Sharing ───────────────────────────────────────────────────────────

@app.post("/api/v2/share", tags=["Sharing"])
async def create_share_endpoint(request: Request):
    """Create a public share link for a model."""
    from .sharing import create_share
    api_key = request.headers.get("X-API-Key", "")
    key_obj = validate_api_key(api_key) if api_key else None
    
    body = await request.json()
    model_id = body.get("model_id")
    if not model_id:
        raise HTTPException(400, "model_id required")
    
    share = create_share(
        model_id=model_id,
        user_id=key_obj.user_id if key_obj else "anonymous",
        title=body.get("title", ""),
        description=body.get("description", ""),
    )
    return JSONResponse({
        "share_id": share.share_id,
        "url": f"/s/{share.share_id}",
        "embed": f'<iframe src="/embed/{share.share_id}" width="600" height="400"></iframe>',
    })


@app.get("/api/v2/gallery", tags=["Sharing"])
async def public_gallery(limit: int = 20):
    """List publicly shared models."""
    from .sharing import list_public_shares
    return JSONResponse({"models": list_public_shares(limit)})


@app.get("/s/{share_id}", tags=["Sharing"])
async def view_shared_model(share_id: str):
    """View a shared 3D model (public page)."""
    from .sharing import get_share
    share = get_share(share_id)
    if not share:
        raise HTTPException(404, "Share not found")
    
    # Return preview page with model loaded
    from .model_store import get_model
    model = get_model(share.get("model_id", ""))
    preview_url = ""
    if model and model.get("output_path"):
        preview_url = f"/preview?url={model['output_path']}"
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>{share.get('title','3D Model')} — PrintForge</title>
<meta property="og:title" content="{share.get('title','3D Model')}">
<meta property="og:description" content="{share.get('description','Generated with PrintForge')}">
<style>body{{margin:0;background:#0d1117;color:#fff;font-family:sans-serif;display:flex;flex-direction:column;align-items:center;padding:40px}}
h1{{font-weight:300;margin-bottom:8px}}p{{color:#8b949e}}
.stats{{display:flex;gap:20px;margin:16px 0;color:#8b949e}}
iframe{{border:1px solid #30363d;border-radius:8px;margin:20px 0}}</style></head>
<body>
<h1>{share.get('title','3D Model')}</h1>
<p>{share.get('description','')}</p>
<div class="stats">
<span>👁 {share.get('views',0)} views</span>
<span>❤️ {share.get('likes',0)} likes</span>
<span>⬇️ {share.get('downloads',0)} downloads</span>
</div>
<iframe src="/preview" width="800" height="500" allowfullscreen></iframe>
<p style="margin-top:20px"><a href="/" style="color:#58a6ff">← Back to PrintForge</a></p>
</body></html>""")


@app.get("/embed/{share_id}", tags=["Sharing"])
async def embed_model(share_id: str):
    """Embeddable 3D viewer for shared models."""
    from .sharing import get_share
    share = get_share(share_id)
    if not share:
        raise HTTPException(404, "Share not found")
    
    from fastapi.responses import HTMLResponse
    # Return minimal Three.js preview page
    return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>PrintForge Embed</title>
<style>*{{margin:0}}body{{background:#0d1117}}</style></head>
<body><script>window.location='/preview'</script></body></html>""")


@app.post("/api/v2/share/{share_id}/like", tags=["Sharing"])
async def like_endpoint(share_id: str):
    from .sharing import like_share
    if not like_share(share_id):
        raise HTTPException(404, "Share not found")
    return JSONResponse({"status": "liked"})


# ── Materials ─────────────────────────────────────────────────────────

@app.get("/api/v2/materials", tags=["Tools"])
async def list_materials_endpoint():
    """List available 3D printing materials with properties."""
    from .material_db import list_materials
    return JSONResponse({"materials": list_materials()})


@app.post("/api/v2/materials/estimate", tags=["Tools"])
async def estimate_cost_endpoint(request: Request):
    """Estimate material cost for a given volume."""
    from .material_db import estimate_material_cost
    body = await request.json()
    volume = body.get("volume_cm3", 0)
    material = body.get("material", "pla")
    return JSONResponse(estimate_material_cost(volume, material))


# ── Mesh Converter ────────────────────────────────────────────────────

@app.post("/api/v2/convert", tags=["Tools"])
async def convert_endpoint(
    mesh_file: UploadFile = File(...),
    target_format: str = "glb",
    target_faces: int = 0,
    color: str = "#4CAF50",
):
    """Convert mesh between formats (STL/OBJ/3MF/GLB/PLY)."""
    from .converter import convert_mesh

    suffix = Path(mesh_file.filename or "model.stl").suffix or ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
        content = await mesh_file.read()
        tmp_in.write(content)
        input_path = tmp_in.name

    out_suffix = f".{target_format}"
    with tempfile.NamedTemporaryFile(suffix=out_suffix, delete=False) as tmp_out:
        output_path = tmp_out.name

    try:
        stats = convert_mesh(
            input_path, output_path,
            target_faces=target_faces if target_faces > 0 else None,
            color=color,
        )
        return FileResponse(
            output_path,
            media_type="application/octet-stream",
            filename=f"converted{out_suffix}",
            headers={
                "X-PrintForge-Input-Faces": str(stats["input_faces"]),
                "X-PrintForge-Output-Faces": str(stats["output_faces"]),
                "X-PrintForge-Simplified": str(stats["simplified"]),
            },
        )
    except Exception as e:
        raise HTTPException(500, f"Conversion failed: {e}")


@app.post("/api/v2/mesh-info", tags=["Tools"])
async def mesh_info_endpoint(mesh_file: UploadFile = File(...)):
    """Get detailed info about a mesh file."""
    from .converter import get_mesh_info

    suffix = Path(mesh_file.filename or "model.stl").suffix or ".stl"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await mesh_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        info = get_mesh_info(tmp_path)
        return JSONResponse(info)
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {e}")


# ── SSE Events ────────────────────────────────────────────────────────

from starlette.responses import StreamingResponse

@app.get("/api/v2/events", tags=["Events"])
async def sse_events(request: Request):
    """Server-Sent Events stream for real-time generation updates.
    
    Connect with: curl -N http://localhost:8000/api/v2/events
    
    Events: generation.started, generation.progress, generation.done, generation.failed, queue.update
    """
    async def event_stream():
        q = event_bus.subscribe()
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            event_bus.unsubscribe(q)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


# ── Batch Generation ──────────────────────────────────────────────────

@app.post("/api/v2/generate/batch", tags=["Generation"])
async def generate_batch(
    request: Request,
    images: List[UploadFile] = File(...),
    format: str = "stl",
    backend: str = "auto",
    multi_view: bool = False,
):
    """Submit multiple images for batch async generation. Returns list of task_ids."""
    api_key_raw = request.headers.get("X-API-Key", "")
    api_key_obj = validate_api_key(api_key_raw) if api_key_raw else None

    if len(images) > 20:
        raise HTTPException(400, "Maximum 20 images per batch")

    queue = GenerationQueue()
    await queue.ensure_workers()

    task_ids = []
    for img in images:
        suffix = Path(img.filename or "upload.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await img.read()
            tmp.write(content)
            input_path = tmp.name

        task_id = queue.submit(
            image_path=input_path,
            output_format=format,
            backend=backend,
            multi_view=multi_view,
            user_id=api_key_obj.user_id if api_key_obj else "anonymous",
            api_key=api_key_raw,
        )
        task_ids.append(task_id)

    return JSONResponse({
        "task_ids": task_ids,
        "count": len(task_ids),
        "queue_size": queue.queue_size,
    })


# ── Webhooks ──────────────────────────────────────────────────────────

@app.post("/api/v2/webhooks", tags=["Webhooks"])
async def add_webhook(request: Request):
    """Register a webhook URL for generation events."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(401, "X-API-Key required")
    key_obj = validate_api_key(api_key)
    if not key_obj:
        raise HTTPException(401, "Invalid API key")
    
    body = await request.json()
    url = body.get("url")
    if not url:
        raise HTTPException(400, "url required")
    
    register_webhook(key_obj.user_id, url, body.get("secret", ""), body.get("events"))
    return JSONResponse({"status": "registered", "url": url})


@app.get("/api/v2/webhooks", tags=["Webhooks"])
async def get_webhooks(request: Request):
    """List registered webhooks."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(401, "X-API-Key required")
    key_obj = validate_api_key(api_key)
    if not key_obj:
        raise HTTPException(401, "Invalid API key")
    
    return JSONResponse({"webhooks": list_webhooks(key_obj.user_id)})


@app.delete("/api/v2/webhooks", tags=["Webhooks"])
async def remove_webhook(request: Request):
    """Remove a webhook."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(401, "X-API-Key required")
    key_obj = validate_api_key(api_key)
    if not key_obj:
        raise HTTPException(401, "Invalid API key")
    
    body = await request.json()
    url = body.get("url")
    if not url:
        raise HTTPException(400, "url required")
    
    if not unregister_webhook(key_obj.user_id, url):
        raise HTTPException(404, "Webhook not found")
    return JSONResponse({"status": "removed"})


# ── Async Generation ──────────────────────────────────────────────────

@app.post("/api/v2/generate/async", tags=["Generation"])
async def generate_async(
    request: Request,
    image: UploadFile = File(...),
    format: str = "stl",
    size_mm: float = 50.0,
    add_base: bool = False,
    backend: str = "auto",
    multi_view: bool = False,
):
    """Submit async generation job. Returns task_id immediately."""
    api_key_raw = request.headers.get("X-API-Key", "")
    api_key_obj = validate_api_key(api_key_raw) if api_key_raw else None
    if api_key_raw and not api_key_obj:
        raise HTTPException(401, "Invalid API key")
    if api_key_obj and api_key_obj.quota_exhausted:
        raise HTTPException(429, "Quota exhausted")

    # Save uploaded image
    suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await image.read()
        tmp.write(content)
        input_path = tmp.name

    queue = GenerationQueue()
    await queue.ensure_workers()

    task_id = queue.submit(
        image_path=input_path,
        output_format=format,
        backend=backend,
        multi_view=multi_view,
        scale_mm=size_mm,
        add_base=add_base,
        user_id=api_key_obj.user_id if api_key_obj else "anonymous",
        api_key=api_key_raw,
    )

    return JSONResponse({
        "task_id": task_id,
        "status": "pending",
        "queue_size": queue.queue_size,
    })


@app.get("/api/v2/generate/{task_id}/status", tags=["Generation"])
async def get_generation_status(task_id: str):
    """Check async generation task status."""
    queue = GenerationQueue()
    status = queue.get_status(task_id)
    if not status:
        raise HTTPException(404, "Task not found")
    return JSONResponse(status)


@app.get("/api/v2/generate/{task_id}/result", tags=["Generation"])
async def get_generation_result(task_id: str):
    """Download result of completed async generation."""
    queue = GenerationQueue()
    task = queue.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    if task.status.value != "done":
        raise HTTPException(409, f"Task not done yet (status: {task.status.value})")
    if not task.result_path or not Path(task.result_path).exists():
        raise HTTPException(404, "Result file not found")
    
    return FileResponse(
        task.result_path,
        media_type="application/octet-stream",
        filename=f"{task_id}.{task.output_format}",
    )


@app.get("/api/v2/queue", tags=["Generation"])
async def queue_status():
    """Get async generation queue status."""
    queue = GenerationQueue()
    return JSONResponse({
        "queue_size": queue.queue_size,
        "active": queue.active_count,
        "recent": queue.list_tasks(limit=10),
    })


# ── Model Management ──────────────────────────────────────────────────

@app.get("/api/v2/models", tags=["Models"])
async def list_user_models(request: Request, limit: int = 50):
    """List generated models for the authenticated user."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        # No auth — return all models
        return JSONResponse({"models": list_models(limit=limit)})
    
    key_obj = validate_api_key(api_key)
    if not key_obj:
        raise HTTPException(401, "Invalid API key")
    
    return JSONResponse({"models": list_models(user_id=key_obj.user_id, limit=limit)})


@app.get("/api/v2/models/{model_id}", tags=["Models"])
async def get_model_detail(model_id: str):
    """Get details for a specific generated model."""
    model = get_model(model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    return JSONResponse(model)


@app.get("/api/v2/models/{model_id}/download", tags=["Models"])
async def download_model(model_id: str):
    """Download a generated model file."""
    model = get_model(model_id)
    if not model:
        raise HTTPException(404, "Model not found")
    
    path = model.get("output_path", "")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Model file not found on disk")
    
    return FileResponse(
        path,
        media_type="application/octet-stream",
        filename=f"{model_id}.{model.get('output_format', 'stl')}",
    )


@app.delete("/api/v2/models/{model_id}", tags=["Models"])
async def delete_model_endpoint(model_id: str, request: Request):
    """Delete a generated model."""
    if not delete_model(model_id):
        raise HTTPException(404, "Model not found")
    return JSONResponse({"status": "deleted", "model_id": model_id})


@app.get("/api/v2/stats", tags=["Models"])
async def user_stats(request: Request):
    """Get generation stats for the authenticated user."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(401, "X-API-Key required")
    
    key_obj = validate_api_key(api_key)
    if not key_obj:
        raise HTTPException(401, "Invalid API key")
    
    return JSONResponse(get_user_stats(key_obj.user_id))


# ── Metrics & Health ─────────────────────────────────────────────────

@app.get("/metrics", tags=["Ops"])
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    m = get_metrics()
    return Response(
        content=m.get_metrics_text(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/health/detail", tags=["Ops"])
async def health_detail():
    """Detailed health check with subsystem status."""
    checker = get_health_checker()
    return JSONResponse(checker.check_all())


# ── 3D Preview route ────────────────────────────────────────────────

@app.get("/api-keys", tags=["UI"])
async def api_keys_page():
    """API key management page."""
    keys_path = os.path.join(_web_dir, "api-keys.html") if os.path.isdir(_web_dir) else None
    if keys_path and os.path.exists(keys_path):
        from fastapi.responses import HTMLResponse
        with open(keys_path) as f:
            return HTMLResponse(f.read())
    raise HTTPException(404, "API keys page not found")


@app.get("/dashboard", tags=["UI"])
async def dashboard_page():
    """Admin dashboard with real-time stats."""
    dash_path = os.path.join(_web_dir, "dashboard.html") if os.path.isdir(_web_dir) else None
    if dash_path and os.path.exists(dash_path):
        from fastapi.responses import HTMLResponse
        with open(dash_path) as f:
            return HTMLResponse(f.read())
    raise HTTPException(404, "Dashboard not found")


@app.get("/preview", tags=["UI"])
async def preview_page():
    """Serve the Three.js 3D preview page."""
    preview_path = os.path.join(_web_dir, "preview.html") if os.path.isdir(_web_dir) else None
    if preview_path and os.path.exists(preview_path):
        from fastapi.responses import HTMLResponse
        with open(preview_path) as f:
            return HTMLResponse(f.read())
    raise HTTPException(404, "Preview page not found")

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


# ── Slicing Endpoint ───────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import List


class SliceResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "ok"})
    gcode_path: str = Field(..., description="Path to generated G-code file")
    print_time_seconds: float = Field(..., description="Estimated print time in seconds")
    print_time_formatted: str = Field(..., description="Human-readable print time")
    filament_grams: float = Field(..., description="Filament weight in grams")
    filament_meters: float = Field(..., description="Filament length in meters")
    layer_count: int = Field(..., description="Number of print layers")
    slicer_used: str = Field(..., description="Slicer engine used")
    slicer_version: str = Field(default="", description="Slicer version")
    hotend_temp_c: int = Field(default=0, description="Hotend temperature in Celsius")
    bed_temp_c: int = Field(default=0, description="Bed temperature in Celsius")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated material cost in USD")
    available_slicers: List[str] = Field(default_factory=list)


class SlicerListResponse(BaseModel):
    slicers: List[dict]


@app.get(
    "/api/v2/slicers",
    tags=["Printers"],
    summary="List detected slicers",
    description="Returns all detected slicer installations (PrusaSlicer, Cura, BambuStudio) with availability status.",
    response_model=SlicerListResponse,
)
async def list_slicers():
    """List all detected slicer installations."""
    from .slicer_bridge import detect_slicers
    detected = detect_slicers()
    return JSONResponse({
        "slicers": [
            {
                "name": s.name,
                "version": s.version,
                "executable": s.executable,
                "is_available": s.is_available,
            }
            for s in detected
        ]
    })


@app.post(
    "/api/v2/slice",
    tags=["Printers"],
    summary="Slice an STL/3MF file to G-code",
    description="Upload an STL or 3MF mesh file and generate G-code using a local slicer (PrusaSlicer, Cura, or BambuStudio). Returns print time estimates, material usage, and layer count.",
    response_model=SliceResponse,
    responses={
        200: {"description": "G-code generated successfully"},
        400: {"description": "Invalid file format"},
        500: {"description": "Slicing failed"},
    },
)
async def slice_file(
    mesh_file: UploadFile = File(...),
    printer: str = Form("bambu_a1"),
    material: str = Form("pla"),
    layer_height: float = Form(0.2),
    infill: float = Form(0.15),
    support: bool = Form(False),
    slicer: str = Form("auto"),
):
    """Slice an STL/3MF file to G-code."""
    from .slicer_bridge import (
        slice_mesh,
        SliceConfig,
        detect_slicers,
        format_duration,
        MATERIAL_PROFILES,
    )

    # Validate file type
    allowed_exts = {".stl", ".3mf", ".obj"}
    suffix = Path(mesh_file.filename or "model.stl").suffix.lower()
    if suffix not in allowed_exts:
        raise HTTPException(400, f"Unsupported file format: {suffix}. Supported: {', '.join(allowed_exts)}")

    # Save uploaded mesh
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await mesh_file.read()
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(400, "File too large (max 100MB)")
        tmp.write(content)
        mesh_path = tmp.name

    # Generate output G-code path
    output_dir = Path(tempfile.gettempdir()) / "printforge_sliced"
    output_dir.mkdir(exist_ok=True)
    output_gcode = str(output_dir / f"{Path(mesh_file.filename).stem}.gcode")

    try:
        # Build slice config
        config = SliceConfig(
            printer_profile=printer,
            material=material,
            layer_height=layer_height,
            infill=infill,
            support=support,
        )

        # Run slicing
        result = slice_mesh(
            mesh_path=mesh_path,
            output_gcode=output_gcode,
            config=config,
            slicer=slicer,
        )

        # Get available slicers for response
        available = [s.name for s in detect_slicers() if s.is_available]

        return JSONResponse({
            "status": "ok",
            "gcode_path": result.gcode_path,
            "print_time_seconds": round(result.print_time_seconds, 1),
            "print_time_formatted": format_duration(result.print_time_seconds),
            "filament_grams": round(result.filament_grams, 2),
            "filament_meters": round(result.filament_meters, 3),
            "layer_count": result.layer_count,
            "slicer_used": result.slicer_used,
            "slicer_version": result.slicer_version,
            "hotend_temp_c": result.hotend_temp_c,
            "bed_temp_c": result.bed_temp_c,
            "estimated_cost_usd": round(result.estimated_cost_usd, 4),
            "available_slicers": available,
        })
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Slicing failed")
        raise HTTPException(500, f"Slicing failed: {str(e)}")
    finally:
        # Clean up temp mesh file
        Path(mesh_path).unlink(missing_ok=True)


# ── Print Farm Management ───────────────────────────────────────────────

class PrinterRegisterRequest(BaseModel):
    name: str = Field(..., description="Printer display name")
    model: str = Field(..., description="Printer model (e.g., Bambu X1C)")
    ip_address: str = Field(..., description="Printer IP address")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class PrinterRegisterResponse(BaseModel):
    status: str
    printer: Dict[str, Any]


class TaskAssignRequest(BaseModel):
    job_id: str = Field(..., description="Job ID to assign")
    filename: str = Field(..., description="File name to print")
    material: str = Field(default="PLA", description="Filament material")
    estimated_time_min: int = Field(default=60, description="Estimated print time in minutes")


class TaskAssignResponse(BaseModel):
    status: str
    task: Dict[str, Any]
    printer: Dict[str, Any]


@app.post(
    "/api/v2/printers/register",
    tags=["Print Farm"],
    summary="Register a printer",
    response_model=PrinterRegisterResponse,
)
async def register_printer(req: PrinterRegisterRequest):
    """Register a new printer in the print farm."""
    try:
        printer = print_farm.register_printer(
            name=req.name,
            model=req.model,
            ip_address=req.ip_address,
            metadata=req.metadata or {},
        )
        return JSONResponse({
            "status": "registered",
            "printer": printer.to_dict(),
        })
    except Exception as e:
        logger.exception("Failed to register printer")
        raise HTTPException(500, str(e))


@app.get(
    "/api/v2/printers",
    tags=["Print Farm"],
    summary="List all printers",
)
async def list_printers():
    """Get all registered printers in the print farm."""
    return JSONResponse({
        "status": "ok",
        "farm_stats": print_farm.get_farm_stats(),
        "printers": [p.to_dict() for p in print_farm.list_printers()],
    })


@app.post(
    "/api/v2/printers/{printer_id}/assign",
    tags=["Print Farm"],
    summary="Assign task to printer",
    response_model=TaskAssignResponse,
)
async def assign_task_to_printer(printer_id: str, req: TaskAssignRequest):
    """Manually assign a print task to a specific printer."""
    task = print_farm.assign_task(
        printer_id=printer_id,
        job_id=req.job_id,
        filename=req.filename,
        material=req.material,
        estimated_time_min=req.estimated_time_min,
    )
    if not task:
        raise HTTPException(404, "Printer not found or not available")
    
    printer = print_farm.get_printer(printer_id)
    return JSONResponse({
        "status": "assigned",
        "task": asdict(task),
        "printer": printer.to_dict(),
    })


@app.post(
    "/api/v2/printers/{printer_id}/complete",
    tags=["Print Farm"],
)
async def complete_print_task(
    printer_id: str,
    success: bool = Query(..., description="Whether the task succeeded"),
    error_message: Optional[str] = Query(None, description="Error message if failed"),
):
    """Mark a printer's current task as completed."""
    if not print_farm.complete_task(printer_id, success, error_message):
        raise HTTPException(404, "Printer or task not found")
    return JSONResponse({"status": "completed", "printer": print_farm.get_printer(printer_id).to_dict()})


@app.post(
    "/api/v2/printers/{printer_id}/status",
    tags=["Print Farm"],
)
async def update_printer_status(
    printer_id: str,
    status: str = Query(..., description="New status: idle, offline, error, maintenance"),
):
    """Update a printer's status."""
    try:
        new_status = PrinterStatus(status)
    except ValueError:
        raise HTTPException(400, f"Invalid status. Must be one of: {[s.value for s in PrinterStatus]}")
    
    printer = print_farm.update_printer_status(printer_id, new_status)
    if not printer:
        raise HTTPException(404, "Printer not found")
    return JSONResponse({"status": "updated", "printer": printer.to_dict()})


@app.post(
    "/api/v2/printers/auto-assign",
    tags=["Print Farm"],
    response_model=TaskAssignResponse,
)
async def auto_assign_task(req: TaskAssignRequest):
    """Automatically assign a task to the best available printer."""
    result = print_farm.auto_assign_task(
        job_id=req.job_id,
        filename=req.filename,
        material=req.material,
        estimated_time_min=req.estimated_time_min,
    )
    if not result:
        raise HTTPException(503, "No available printers")
    
    printer, task = result
    return JSONResponse({
        "status": "assigned",
        "task": asdict(task),
        "printer": printer.to_dict(),
    })
