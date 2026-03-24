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

    # ── printforge split <mesh> ─────────────────────────────────────
    p_split = subparsers.add_parser("split", help="Split large mesh into printable parts")
    p_split.add_argument("mesh", help="Input mesh file (STL/OBJ/3MF)")
    p_split.add_argument("--volume", default="256x256x256", help="Build volume as XxYxZ in mm")
    p_split.add_argument("-o", "--output", help="Output directory for split parts")
    p_split.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    p_split.set_defaults(func=cmd_split)

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
