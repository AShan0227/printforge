#!/usr/bin/env python3
"""
Generate 5 demo images for PrintForge README and MakerWorld uploads.
Uses PIL to create simple recognizable shapes on white backgrounds.

Usage: python3 scripts/generate_demo_images.py
"""

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "examples"
SIZE = 512
BG = (255, 255, 255)


def draw_cup(draw: ImageDraw.ImageDraw):
    """Simple cup/mug silhouette."""
    # Cup body
    draw.rectangle([140, 120, 340, 400], fill=(70, 130, 180))
    # Cup rim (slightly wider)
    draw.rectangle([130, 110, 350, 140], fill=(60, 110, 160))
    # Handle
    draw.arc([330, 180, 420, 320], start=-60, end=60, fill=(60, 110, 160), width=18)
    # Base
    draw.rectangle([130, 390, 350, 410], fill=(50, 90, 140))


def draw_star(draw: ImageDraw.ImageDraw):
    """Five-pointed star."""
    cx, cy, r_outer, r_inner = 256, 256, 180, 72
    points = []
    for i in range(10):
        angle = math.radians(i * 36 - 90)
        r = r_outer if i % 2 == 0 else r_inner
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=(218, 165, 32))
    draw.polygon(points, outline=(180, 130, 20), width=3)


def draw_logo(draw: ImageDraw.ImageDraw):
    """'PF' text logo."""
    # Background circle
    draw.ellipse([100, 100, 412, 412], fill=(30, 30, 30))
    # Try to use a bold font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 180)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("arial.ttf", 180)
        except (OSError, IOError):
            font = ImageFont.load_default()
    draw.text((140, 140), "PF", fill=(0, 200, 255), font=font)


def draw_gear(draw: ImageDraw.ImageDraw):
    """Gear/cog shape."""
    cx, cy = 256, 256
    teeth = 12
    r_outer, r_inner, r_hole = 190, 140, 50
    points = []
    for i in range(teeth * 2):
        angle = math.radians(i * 360 / (teeth * 2) - 90)
        r = r_outer if i % 2 == 0 else r_inner
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=(120, 120, 130))
    draw.polygon(points, outline=(80, 80, 90), width=2)
    # Center hole
    draw.ellipse([cx - r_hole, cy - r_hole, cx + r_hole, cy + r_hole], fill=BG)


def draw_heart(draw: ImageDraw.ImageDraw):
    """Heart shape using parametric curve."""
    cx, cy, scale = 256, 270, 12
    points = []
    for deg in range(360):
        t = math.radians(deg)
        x = scale * 16 * math.sin(t) ** 3
        y = -scale * (13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        points.append((cx + x, cy + y))
    draw.polygon(points, fill=(220, 40, 60))
    draw.polygon(points, outline=(180, 30, 50), width=2)


SHAPES = {
    "demo_cup": draw_cup,
    "demo_star": draw_star,
    "demo_logo": draw_logo,
    "demo_gear": draw_gear,
    "demo_heart": draw_heart,
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, draw_fn in SHAPES.items():
        img = Image.new("RGB", (SIZE, SIZE), BG)
        draw = ImageDraw.Draw(img)
        draw_fn(draw)
        out_path = OUTPUT_DIR / f"{name}.png"
        img.save(out_path, "PNG")
        print(f"  Generated: {out_path}")
    print(f"\nDone — {len(SHAPES)} demo images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
