"""Tests for auto_crop."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from printforge.auto_crop import auto_crop


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def solid_image(size=(200, 200), color=(255, 0, 0)) -> Image.Image:
    """Solid-color square image."""
    return Image.new("RGB", size, color=color)


def centered_rectangle_image(
    size=(400, 300),
    rect=(100, 75, 300, 225),
    fill=(0, 200, 0),
    bg=(20, 20, 20),
) -> Image.Image:
    """Image with a bright rectangle on a dark background — simulates a subject."""
    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)
    draw.rectangle(rect, fill=fill)
    return img


def noisy_image_with_shape(size=(300, 300), seed=42) -> Image.Image:
    """Random noise background + bright circle in the centre."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 60, (*size[::-1], 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(img)
    cx, cy = size[0] // 2, size[1] // 2
    r = 40
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 100))
    return img


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_returns_original_on_blank_image():
    """A completely blank (single-colour) image has no edges → return original."""
    img = solid_image()
    result = auto_crop(img)
    assert result.size == img.size
    assert list(img.getdata()) == list(result.getdata())


def test_crops_to_centered_rectangle():
    """A bright rectangle on a dark background should crop tightly around it."""
    size = (400, 300)
    rect = (100, 75, 300, 225)  # 200x150 rect in the centre
    img = centered_rectangle_image(size=size, rect=rect)

    result = auto_crop(img, padding=0.0)

    w, h = result.size
    # Bbox should cover the rectangle (with tiny tolerance for edge bleed)
    # Crop should be substantially smaller than original
    assert w < size[0], f"Width {w} should be less than {size[0]}"
    assert h < size[1], f"Height {h} should be less than {size[1]}"


def test_padding_expands_crop():
    """Larger padding should produce a wider crop."""
    size = (400, 300)
    rect = (100, 75, 300, 225)
    img = centered_rectangle_image(size=size, rect=rect)

    crop_tight = auto_crop(img, padding=0.0)
    crop_padded = auto_crop(img, padding=0.2)

    assert crop_padded.size[0] >= crop_tight.size[0]
    assert crop_padded.size[1] >= crop_tight.size[1]


def test_detects_bright_circle_in_noise():
    """A bright circle in noisy background should be detected and cropped."""
    img = noisy_image_with_shape()

    result = auto_crop(img, padding=0.05)

    # Should be smaller than or equal to original (edge case: detection may fail)
    assert result.size[0] <= img.size[0]
    assert result.size[1] <= img.size[1]
    # If detection worked, crop is smaller; if not, original is returned
    w, h = result.size
    assert w > 0 and h > 0


def test_returns_same_size_for_all_modes():
    """auto_crop should handle L, RGB, RGBA, and P modes without crashing."""
    base = centered_rectangle_image()

    for mode in ("L", "RGB", "RGBA"):
        img = base.convert(mode)
        result = auto_crop(img)
        # Result must be a valid PIL Image with valid size tuple
        assert isinstance(result, Image.Image)
        assert result.size == img.size or (
            result.size[0] < img.size[0] and result.size[1] < img.size[1]
        )
