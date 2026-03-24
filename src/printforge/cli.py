"""PrintForge CLI — One photo to 3D print."""

import argparse
import logging
import sys
from pathlib import Path

from .pipeline import PrintForgePipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(
        prog="printforge",
        description="🏭 PrintForge — One photo to 3D print",
    )
    parser.add_argument("image", help="Input image path (jpg/png)")
    parser.add_argument("-o", "--output", help="Output file path (default: <image>.3mf)")
    parser.add_argument("--format", choices=["3mf", "stl"], default="3mf", help="Output format")
    parser.add_argument("--size", type=float, default=50.0, help="Target size in mm (default: 50)")
    parser.add_argument("--resolution", type=int, default=256, help="Mesh resolution (default: 256)")
    parser.add_argument("--add-base", action="store_true", help="Add flat base for easier printing")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Inference device")
    parser.add_argument("--max-faces", type=int, default=200000, help="Max face count")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Validate input
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = str(image_path.with_suffix(f".{args.format}"))
    
    # Configure pipeline
    config = PipelineConfig(
        device=args.device,
        mc_resolution=args.resolution,
        scale_mm=args.size,
        max_faces=args.max_faces,
        add_base=args.add_base,
        output_format=args.format,
    )
    
    # Run
    print(f"🏭 PrintForge v0.1.0")
    print(f"   Input:  {image_path}")
    print(f"   Output: {output_path}")
    print(f"   Size:   {args.size}mm")
    print(f"   Device: {args.device}")
    print()
    
    pipeline = PrintForgePipeline(config)
    result = pipeline.run(str(image_path), output_path)
    
    # Report
    print()
    print(f"{'='*50}")
    print(f"✅ Done!")
    print(f"   File:       {result.mesh_path}")
    print(f"   Vertices:   {result.vertices:,}")
    print(f"   Faces:      {result.faces:,}")
    print(f"   Watertight: {'✅' if result.is_watertight else '⚠️ No'}")
    print(f"   Wall OK:    {'✅' if result.wall_thickness_ok else '⚠️ Too thin'}")
    print(f"   Time:       {result.duration_ms:.0f}ms")
    
    if result.warnings:
        print()
        print("⚠️  Warnings:")
        for w in result.warnings:
            print(f"   - {w}")
    
    print()
    print(f"Next: Open {output_path} in Bambu Studio → Slice → Print 🖨️")


if __name__ == "__main__":
    main()
