"""PrintForge CLI — One photo to 3D print, and more."""

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import PrintForgePipeline, PipelineConfig


def _add_common_args(parser):
    """Add common arguments shared across subcommands."""
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("--format", choices=["3mf", "stl", "obj"], default="3mf", help="Output format")
    parser.add_argument("--size", type=float, default=50.0, help="Target size in mm (default: 50)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Inference device")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")


def _setup_logging(verbose: bool):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_image(args):
    """Image-to-3D command (original pipeline)."""
    _setup_logging(args.verbose)

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(image_path.with_suffix(f".{args.format}"))

    config = PipelineConfig(
        device=args.device,
        mc_resolution=args.resolution,
        scale_mm=args.size,
        max_faces=args.max_faces,
        add_base=args.add_base,
        output_format=args.format,
    )

    print(f"PrintForge v0.2.0")
    print(f"  Input:  {image_path}")
    print(f"  Output: {output_path}")
    print(f"  Size:   {args.size}mm")
    print()

    pipeline = PrintForgePipeline(config)
    result = pipeline.run(str(image_path), output_path)
    _print_result(result)


def cmd_text(args):
    """Text-to-3D command."""
    _setup_logging(args.verbose)

    from .text_to_3d import TextTo3DPipeline, TextTo3DConfig

    description = args.description
    output_path = args.output or f"printforge_output.{args.format}"

    image_path = getattr(args, "image", None)

    config = TextTo3DConfig()
    pipeline_config = PipelineConfig(
        device=args.device,
        scale_mm=args.size,
        output_format=args.format,
    )

    print(f"PrintForge v0.2.0 — Text to 3D")
    print(f"  Description: {description}")
    print(f"  Output:      {output_path}")
    print()

    text_pipeline = TextTo3DPipeline(config)
    result = text_pipeline.run(
        description=description,
        output_path=output_path,
        image_path=image_path,
        save_prompt=args.save_prompt if hasattr(args, "save_prompt") else None,
        pipeline_config=pipeline_config,
    )

    if result.used_fallback:
        print(f"Image generation unavailable.")
        print(f"Generated prompt: {result.prompt_used}")
        print(f"Provide your own image with: printforge image <your-image> -o {output_path}")
    elif result.pipeline_result:
        _print_result(result.pipeline_result)


def cmd_optimize(args):
    """Optimize/analyze a mesh for printing."""
    _setup_logging(args.verbose)
    import trimesh
    from .print_optimizer import PrintOptimizer, PRINTER_PRESETS

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"Error: File not found: {mesh_path}", file=sys.stderr)
        sys.exit(1)

    mesh = trimesh.load(str(mesh_path))

    preset = PRINTER_PRESETS.get(args.printer, PRINTER_PRESETS["bambu-a1"])
    build_vol = preset["volume"]

    optimizer = PrintOptimizer()

    print(f"PrintForge v0.2.0 — Print Optimizer")
    print(f"  Model:   {mesh_path}")
    print(f"  Printer: {preset['name']}")
    print(f"  Size:    {mesh.bounding_box.extents[0]:.1f} x {mesh.bounding_box.extents[1]:.1f} x {mesh.bounding_box.extents[2]:.1f}mm")
    print()

    # Orientation
    orientation = optimizer.find_best_orientation(mesh)
    print(f"Best orientation:")
    print(f"  Height:   {orientation.height:.1f}mm")
    print(f"  Support:  ~{orientation.support_volume_estimate:.0f}mm^3")
    print(f"  Base:     {orientation.base_area:.1f}mm^2")
    print()

    # Estimates
    estimate = optimizer.estimate_material(
        mesh,
        infill=args.infill,
        layer_height=args.layer_height,
        material=args.material,
    )
    print(f"Estimates ({args.material.upper()}, {args.infill*100:.0f}% infill):")
    print(f"  Time:     ~{estimate.print_time_minutes:.0f} min")
    print(f"  Filament: {estimate.filament_grams:.1f}g ({estimate.filament_meters:.2f}m)")
    print(f"  Layers:   {estimate.layer_count}")
    print()

    # Printability
    issues = optimizer.check_printability(mesh, build_volume=build_vol)
    if issues:
        print(f"Issues:")
        for issue in issues:
            icon = {"error": "ERROR", "warning": "WARN", "info": "INFO"}[issue.severity]
            print(f"  [{icon}] {issue.message}")
    else:
        print(f"No printability issues found.")


def cmd_cost(args):
    """Estimate printing cost for a mesh."""
    _setup_logging(args.verbose)
    import trimesh
    from .cost_estimator import CostEstimator

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"Error: File not found: {mesh_path}", file=sys.stderr)
        sys.exit(1)

    mesh = trimesh.load(str(mesh_path))
    infill = args.infill / 100.0 if args.infill > 1 else args.infill

    estimator = CostEstimator()
    est = estimator.estimate(
        mesh,
        material=args.material,
        infill=infill,
        layer_height=args.layer_height,
    )

    print(f"PrintForge — Cost Estimate")
    print(f"  Model:    {mesh_path}")
    print(f"  Material: {args.material.upper()}")
    print(f"  Infill:   {infill*100:.0f}%")
    print(f"  Layer:    {args.layer_height}mm")
    print()
    print(f"Filament:")
    print(f"  Weight:   {est.filament_grams:.1f}g")
    print(f"  Length:   {est.filament_meters:.2f}m")
    print(f"  Cost:     ${est.filament_cost_usd:.2f}")
    print()
    print(f"Print time: {est.print_time_hours:.1f}h")
    print(f"Electricity: ${est.electricity_cost_usd:.2f}")
    print(f"Total cost:  ${est.total_cost_usd:.2f}")


def cmd_split(args):
    """Split a mesh into printable parts."""
    _setup_logging(args.verbose)
    import trimesh
    from .part_splitter import PartSplitter, SplitConfig, BuildVolume

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"Error: File not found: {mesh_path}", file=sys.stderr)
        sys.exit(1)

    mesh = trimesh.load(str(mesh_path))
    build_vol = BuildVolume.from_string(args.volume)

    splitter = PartSplitter(SplitConfig(build_volume=build_vol))

    needs_split, exceeded = splitter.needs_splitting(mesh)
    if not needs_split:
        print(f"Model fits in build volume ({args.volume}). No splitting needed.")
        return

    print(f"PrintForge v0.2.0 — Part Splitter")
    print(f"  Model:  {mesh_path}")
    print(f"  Volume: {args.volume}")
    print(f"  Exceeds: {', '.join(exceeded)} axis")
    print()

    result = splitter.split(mesh)

    output_dir = Path(args.output) if args.output else mesh_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = mesh_path.stem
    for part in result.parts:
        part_path = output_dir / f"{stem}_part{part.part_index}.stl"
        part.mesh.export(str(part_path), file_type="stl")
        bb = part.bounding_box
        print(f"  Part {part.part_index}: {part_path.name}  pins={part.has_pins}  holes={part.has_holes}")

    print()
    print(f"Split into {result.num_parts} parts. All fit: {result.fits_in_volume}")


def cmd_batch(args):
    """Batch process a directory of images to 3D models."""
    _setup_logging(args.verbose)
    from .batch import BatchProcessor

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: Not a directory: {input_dir}", file=sys.stderr)
        sys.exit(1)

    image_paths = BatchProcessor.collect_images(str(input_dir))
    if not image_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or str(input_dir / "output")
    fmt = args.format

    config = PipelineConfig(device=args.device, scale_mm=args.size)
    processor = BatchProcessor(config=config, max_workers=args.workers)

    print(f"PrintForge — Batch Processing")
    print(f"  Input:   {input_dir} ({len(image_paths)} images)")
    print(f"  Output:  {output_dir}")
    print(f"  Format:  {fmt}")
    print(f"  Workers: {args.workers}")
    print()

    def on_progress(done, total, item):
        status = "OK" if item.success else f"FAIL: {item.error}"
        print(f"  [{done}/{total}] {Path(item.input_path).name} — {status}")

    result = processor.process(image_paths, output_dir, fmt, progress_callback=on_progress)
    print()
    print(f"Done: {result.succeeded} succeeded, {result.failed} failed, {result.total_duration_ms:.0f}ms total")


def cmd_quality(args):
    """Score mesh quality for 3D printing."""
    _setup_logging(args.verbose)
    import trimesh
    from .quality import QualityScorer

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"Error: File not found: {mesh_path}", file=sys.stderr)
        sys.exit(1)

    mesh = trimesh.load(str(mesh_path), force="mesh")
    scorer = QualityScorer()
    report = scorer.score(mesh)

    print(f"PrintForge — Quality Score")
    print(f"  Model:  {mesh_path}")
    print(f"  Grade:  {report.grade} ({report.total_score}/100)")
    print()
    print(f"Breakdown:")
    print(f"  Watertight:    {report.watertight_score}/30  ({'Yes' if report.is_watertight else 'No'})")
    print(f"  Face count:    {report.face_count_score}/20  ({report.face_count:,} faces)")
    print(f"  Aspect ratio:  {report.aspect_ratio_score}/15  ({report.aspect_ratio:.1f})")
    print(f"  Thin walls:    {report.thin_wall_score}/20  (min {report.min_thickness_mm:.2f}mm)")
    print(f"  Overhangs:     {report.overhang_score}/15  ({report.overhang_percentage:.1f}%)")


def cmd_repair(args):
    """Repair a broken mesh."""
    _setup_logging(args.verbose)
    import trimesh
    from .repair import MeshRepair

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"Error: File not found: {mesh_path}", file=sys.stderr)
        sys.exit(1)

    mesh = trimesh.load(str(mesh_path), force="mesh")
    output_path = args.output or str(mesh_path.with_stem(mesh_path.stem + "_fixed"))

    repairer = MeshRepair(voxel_resolution=args.resolution)
    repaired, report = repairer.repair(mesh)

    repaired.export(output_path, file_type=Path(output_path).suffix.lstrip(".") or "stl")

    print(f"PrintForge — Mesh Repair")
    print(f"  Input:  {mesh_path}")
    print(f"  Output: {output_path}")
    print()
    print(f"  Watertight: {report.was_watertight_before} → {report.is_watertight_after}")
    print(f"  Faces:      {report.input_faces:,} → {report.output_faces:,}")
    print(f"  Vertices:   {report.input_vertices:,} → {report.output_vertices:,}")
    if report.used_voxel_remesh:
        print(f"  Voxel remesh applied (resolution {args.resolution})")
    print()
    print(f"Repairs performed:")
    for r in report.repairs_performed:
        print(f"  - {r}")


def cmd_benchmark(args):
    """Run performance benchmarks."""
    _setup_logging(args.verbose)
    from .benchmark import BenchmarkSuite

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    suite = BenchmarkSuite()

    print("PrintForge — Performance Benchmark")
    print(f"  Image: {image_path}")
    print()

    # Inference benchmark
    print("Inference benchmarks:")
    results = suite.benchmark_inference(str(image_path), backends=args.backends)
    for r in results:
        if r.error:
            print(f"  {r.backend}: FAILED ({r.error})")
        else:
            print(f"  {r.backend}: {r.duration_ms:.0f}ms, {r.vertices} verts, quality={r.quality_score}")
    print()

    # Full pipeline benchmark
    print("Full pipeline benchmark:")
    report = suite.benchmark_pipeline(str(image_path))
    for stage in report.pipeline_stages:
        print(f"  {stage.stage}: {stage.duration_ms:.1f}ms")
    print(f"  TOTAL: {report.total_pipeline_ms:.0f}ms")
    print()
    print(f"Report saved to: {suite.BENCHMARK_REPORT_PATH}")


def cmd_download_model(args):
    """Download model weights for local inference."""
    _setup_logging(args.verbose)

    model_name = args.model
    models_dir = Path.home() / ".printforge" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_configs = {
        "triposr": {
            "url": "https://huggingface.co/stabilityai/TripoSR/resolve/main/model.ckpt",
            "filename": "triposr_model.ckpt",
            "name": "TripoSR",
        },
        "hunyuan3d": {
            "url": "https://huggingface.co/Tencent/Hunyuan3D-2/resolve/main/model.safetensors",
            "filename": "hunyuan3d_model.safetensors",
            "name": "Hunyuan3D-2",
        },
    }

    cfg = model_configs[model_name]
    dest = models_dir / cfg["filename"]

    if dest.exists():
        print(f"{cfg['name']} already downloaded at {dest}")
        print(f"  Size: {dest.stat().st_size / 1024 / 1024:.1f} MB")
        return

    print(f"Downloading {cfg['name']} to {dest}...")
    print(f"  URL: {cfg['url']}")

    try:
        import requests

        resp = requests.get(cfg["url"], stream=True, timeout=30)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 8192

        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    bar_len = 40
                    filled = int(bar_len * downloaded / total)
                    bar = "█" * filled + "░" * (bar_len - filled)
                    print(f"\r  [{bar}] {pct:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="", flush=True)

        print()
        print(f"Download complete: {dest}")
        print(f"  Size: {dest.stat().st_size / 1024 / 1024:.1f} MB")
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to download server. Check your internet connection.", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        sys.exit(1)


def cmd_predict(args):
    """Predict print failures for a mesh."""
    _setup_logging(args.verbose)
    import trimesh
    from .failure_predictor import FailurePredictor

    mesh_path = Path(args.mesh)
    if not mesh_path.exists():
        print(f"Error: File not found: {mesh_path}", file=sys.stderr)
        sys.exit(1)

    mesh = trimesh.load(str(mesh_path), force="mesh")
    predictor = FailurePredictor()
    prediction = predictor.predict(
        mesh, material=args.material, layer_height=args.layer_height, printer=args.printer,
    )

    print(f"PrintForge — Failure Prediction")
    print(f"  Model:   {mesh_path}")
    print(f"  Printer: {args.printer}")
    print(f"  Risk:    {prediction.risk_level.upper()} ({prediction.risk_score:.0f}/100)")
    print()

    if prediction.risks:
        print(f"Risks identified:")
        for r in prediction.risks:
            icon = {"critical": "CRIT", "high": "HIGH", "medium": "MED", "low": "LOW"}[r.severity]
            print(f"  [{icon}] {r.type}: {r.location}")
            print(f"         → {r.suggestion}")
    else:
        print("No significant risks detected. Model should print successfully.")


def cmd_competitors(args):
    """Show competitor monitoring data."""
    _setup_logging(args.verbose)
    from .competitor_monitor import CompetitorMonitor

    monitor = CompetitorMonitor()

    # Check for updates
    updates = monitor.check_updates()

    print("PrintForge — Competitor Monitor")
    print()

    summary = monitor.get_summary()
    for comp in summary["competitors"]:
        print(f"  {comp['name']} ({comp['url']})")
        print(f"    Version: {comp['version']}")
        print(f"    Features: {', '.join(comp['features'][:3])}...")
        pricing_summary = next(iter(comp['pricing'].values()), "N/A")
        print(f"    Pricing: {pricing_summary}")
        print()

    if updates:
        print(f"Changes detected ({len(updates)}):")
        for u in updates:
            print(f"  [{u.category.upper()}] {u.competitor}: {u.description}")


def cmd_stats(args):
    """Show local usage analytics."""
    _setup_logging(args.verbose)
    from .analytics import Analytics

    analytics = Analytics()
    stats = analytics.get_stats()

    print("PrintForge — Usage Analytics")
    print()
    print(f"  Total events:    {stats['total_events']}")
    print(f"  Generations:     {stats['generations']}")

    if stats['avg_duration_ms']:
        print(f"  Avg duration:    {stats['avg_duration_ms']:.0f}ms")
    if stats['avg_quality_score']:
        print(f"  Avg quality:     {stats['avg_quality_score']:.1f}/100")

    if stats['formats']:
        print()
        print("  Formats:")
        for fmt, count in stats['formats'].items():
            print(f"    {fmt}: {count}")

    if stats['backends']:
        print()
        print("  Backends:")
        for backend, count in stats['backends'].items():
            print(f"    {backend}: {count}")

    if stats['recent_events']:
        print()
        print(f"  Recent events ({len(stats['recent_events'])}):")
        for evt in stats['recent_events'][:5]:
            print(f"    {evt['event_type']} — {evt.get('format', 'N/A')}")


def cmd_video(args):
    """Convert video to 3D model via frame extraction."""
    _setup_logging(args.verbose)
    from .video_to_3d import VideoTo3D

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or str(video_path.with_suffix(f".{args.format}"))

    print(f"PrintForge — Video to 3D")
    print(f"  Input:  {video_path}")
    print(f"  Output: {output_path}")
    print(f"  Frames: {args.frames}")
    print()

    converter = VideoTo3D(num_frames=args.frames)
    result = converter.run(str(video_path), output_path)

    print(f"Extracted {result.num_frames_extracted} frames:")
    for path in result.frame_paths:
        print(f"  {Path(path).name}")
    print()
    if result.mesh_path:
        print(f"Mesh: {result.mesh_path}")
    else:
        print("Multi-view reconstruction: pending (frame extraction only for now)")


def cmd_printers(args):
    """List all supported printer profiles."""
    _setup_logging(args.verbose)
    from .printer_profiles import PRINTER_DB

    print("PrintForge — Supported Printer Profiles")
    print()
    for key in sorted(PRINTER_DB):
        p = PRINTER_DB[key]
        vol = f"{p.build_volume[0]:.0f}x{p.build_volume[1]:.0f}x{p.build_volume[2]:.0f}mm"
        nozzles = ", ".join(f"{n}mm" for n in p.nozzle_sizes)
        auto = "Yes" if p.auto_level else "No"
        print(f"  {key:20s}  {p.name}")
        print(f"    Build volume: {vol}  Speed: {p.max_speed:.0f}mm/s  Auto-level: {auto}")
        print(f"    Nozzles: {nozzles}  Default layer: {p.default_layer_height}mm  Infill: {p.default_infill*100:.0f}%")
        print()


def _print_result(result):
    """Print pipeline result summary."""
    print(f"{'='*50}")
    print(f"Done!")
    print(f"  File:       {result.mesh_path}")
    print(f"  Vertices:   {result.vertices:,}")
    print(f"  Faces:      {result.faces:,}")
    print(f"  Watertight: {'Yes' if result.is_watertight else 'No'}")
    print(f"  Wall OK:    {'Yes' if result.wall_thickness_ok else 'Too thin'}")
    print(f"  Time:       {result.duration_ms:.0f}ms")

    if result.warnings:
        print()
        print("Warnings:")
        for w in result.warnings:
            print(f"  - {w}")


def main():
    parser = argparse.ArgumentParser(
        prog="printforge",
        description="PrintForge — Photo/Text to 3D Print",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── printforge image <path> ─────────────────────────────────────
    p_image = subparsers.add_parser("image", help="Convert an image to a 3D printable model")
    p_image.add_argument("image", help="Input image path (jpg/png)")
    _add_common_args(p_image)
    p_image.add_argument("--resolution", type=int, default=256, help="Mesh resolution")
    p_image.add_argument("--add-base", action="store_true", help="Add flat base")
    p_image.add_argument("--max-faces", type=int, default=200000, help="Max face count")
    p_image.set_defaults(func=cmd_image)

    # ── printforge text '<description>' ─────────────────────────────
    p_text = subparsers.add_parser("text", help="Generate 3D model from text description")
    p_text.add_argument("description", help="Text description of the 3D object")
    _add_common_args(p_text)
    p_text.add_argument("--image", dest="image", help="Provide your own image instead of generating one")
    p_text.add_argument("--save-prompt", help="Save the generated image prompt to a file")
    p_text.set_defaults(func=cmd_text)

    # ── printforge optimize <mesh> ──────────────────────────────────
    p_opt = subparsers.add_parser("optimize", help="Analyze and optimize a mesh for printing")
    p_opt.add_argument("mesh", help="Input mesh file (STL/OBJ/3MF)")
    p_opt.add_argument("--printer", default="bambu-a1", help="Printer preset")
    p_opt.add_argument("--infill", type=float, default=0.15, help="Infill density (0.0-1.0)")
    p_opt.add_argument("--layer-height", type=float, default=0.2, help="Layer height in mm")
    p_opt.add_argument("--material", default="pla", help="Material (pla/petg/abs/tpu)")
    p_opt.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_opt.set_defaults(func=cmd_optimize)

    # ── printforge cost <mesh> ──────────────────────────────────────
    p_cost = subparsers.add_parser("cost", help="Estimate printing cost for a mesh")
    p_cost.add_argument("mesh", help="Input mesh file (STL/OBJ/3MF)")
    p_cost.add_argument("--material", default="PLA", help="Material (PLA/PETG/ABS/TPU)")
    p_cost.add_argument("--infill", type=float, default=20, help="Infill percentage (0-100)")
    p_cost.add_argument("--layer-height", type=float, default=0.2, help="Layer height in mm")
    p_cost.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_cost.set_defaults(func=cmd_cost)

    # ── printforge split <mesh> ─────────────────────────────────────
    p_split = subparsers.add_parser("split", help="Split large mesh into printable parts")
    p_split.add_argument("mesh", help="Input mesh file (STL/OBJ/3MF)")
    p_split.add_argument("--volume", default="256x256x256", help="Build volume as XxYxZ in mm")
    p_split.add_argument("-o", "--output", help="Output directory for split parts")
    p_split.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_split.set_defaults(func=cmd_split)

    # ── printforge batch <input_dir> ──────────────────────────────────
    p_batch = subparsers.add_parser("batch", help="Batch process images to 3D models")
    p_batch.add_argument("input_dir", help="Directory containing images")
    _add_common_args(p_batch)
    p_batch.add_argument("--workers", type=int, default=3, help="Max parallel workers (default: 3)")
    p_batch.set_defaults(func=cmd_batch)

    # ── printforge quality <mesh> ──────────────────────────────────
    p_quality = subparsers.add_parser("quality", help="Score mesh quality for 3D printing")
    p_quality.add_argument("mesh", help="Input mesh file (STL/OBJ/3MF)")
    p_quality.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_quality.set_defaults(func=cmd_quality)

    # ── printforge repair <mesh> ───────────────────────────────────
    p_repair = subparsers.add_parser("repair", help="Repair a broken mesh for printing")
    p_repair.add_argument("mesh", help="Input mesh file (STL/OBJ/3MF)")
    p_repair.add_argument("-o", "--output", help="Output file path")
    p_repair.add_argument("--resolution", type=int, default=128, help="Voxel remesh resolution")
    p_repair.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_repair.set_defaults(func=cmd_repair)

    # ── printforge benchmark <image> ─────────────────────────────────
    p_bench = subparsers.add_parser("benchmark", help="Run performance benchmarks")
    p_bench.add_argument("image", help="Test image for benchmarking")
    p_bench.add_argument("--backends", nargs="*", default=["placeholder", "hunyuan3d"],
                         help="Inference backends to benchmark")
    p_bench.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_bench.set_defaults(func=cmd_benchmark)

    # ── printforge download-model ────────────────────────────────────
    p_dl = subparsers.add_parser("download-model", help="Download model weights for local inference")
    p_dl.add_argument("--model", choices=["hunyuan3d", "triposr"], default="triposr",
                      help="Model to download (default: triposr)")
    p_dl.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_dl.set_defaults(func=cmd_download_model)

    # ── printforge predict <mesh> ──────────────────────────────────
    p_predict = subparsers.add_parser("predict", help="Predict print failures for a mesh")
    p_predict.add_argument("mesh", help="Input mesh file (STL/OBJ/3MF)")
    p_predict.add_argument("--printer", default="a1", help="Printer preset (a1/x1c/p1s/prusa-mk4/ender3)")
    p_predict.add_argument("--material", default="PLA", help="Material (PLA/PETG/ABS/TPU)")
    p_predict.add_argument("--layer-height", type=float, default=0.2, help="Layer height in mm")
    p_predict.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_predict.set_defaults(func=cmd_predict)

    # ── printforge competitors ────────────────────────────────────
    p_comp = subparsers.add_parser("competitors", help="Show competitor monitoring data")
    p_comp.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_comp.set_defaults(func=cmd_competitors)

    # ── printforge stats ──────────────────────────────────────────
    p_stats = subparsers.add_parser("stats", help="Show local usage analytics")
    p_stats.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_stats.set_defaults(func=cmd_stats)

    # ── printforge video <path> ───────────────────────────────────
    p_video = subparsers.add_parser("video", help="Convert video to 3D model via frame extraction")
    p_video.add_argument("video", help="Input video file (mp4/mov/avi)")
    p_video.add_argument("-o", "--output", help="Output file path")
    p_video.add_argument("--format", choices=["3mf", "stl", "obj"], default="3mf", help="Output format")
    p_video.add_argument("--frames", type=int, default=8, help="Number of frames to extract (default: 8)")
    p_video.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_video.set_defaults(func=cmd_video)

    # ── printforge printers ─────────────────────────────────────────
    p_printers = subparsers.add_parser("printers", help="List all supported printer profiles")
    p_printers.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_printers.set_defaults(func=cmd_printers)

    # ── Legacy: printforge <image> (backward compat) ────────────────
    args = parser.parse_args()

    if args.command is None:
        # Check if first positional arg looks like a file (backward compat with v0.1)
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            candidate = Path(sys.argv[1])
            if candidate.exists() and candidate.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                # Reparse as image command
                sys.argv.insert(1, "image")
                args = parser.parse_args()
                args.func(args)
                return

        parser.print_help()
        sys.exit(1)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
