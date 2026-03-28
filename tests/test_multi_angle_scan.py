"""Tests for multi_angle_scan.py — no API calls, use PIL to create test images."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from printforge.multi_angle_scan import (
    MultiAngleScanner,
    ScanSession,
    ViewImage,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_temp_image(color: tuple[int, int, int], size: tuple[int, int] = (200, 300)) -> str:
    """Create a temporary JPEG image with a solid color."""
    img = Image.new("RGB", size, color)
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img.save(path, "JPEG")
    return path


@pytest.fixture
def four_image_paths() -> list[str]:
    """Four test images."""
    paths = [
        make_temp_image((255, 0, 0)),   # red   → front
        make_temp_image((0, 255, 0)),   # green → right
        make_temp_image((0, 0, 255)),   # blue  → back
        make_temp_image((255, 255, 0)),  # yellow → left
    ]
    yield paths
    for p in paths:
        os.unlink(p)


@pytest.fixture
def two_image_paths() -> list[str]:
    """Two test images."""
    paths = [
        make_temp_image((200, 0, 0)),
        make_temp_image((0, 0, 200)),
    ]
    yield paths
    for p in paths:
        os.unlink(p)


# --------------------------------------------------------------------------- #
# Test 1: create_session correctly classifies view angles
# --------------------------------------------------------------------------- #

def test_create_session_angle_classification(four_image_paths: list[str]):
    """With ≥4 images the scanner maps indices 0/1/2/3 → front/right/back/left."""
    scanner = MultiAngleScanner()
    session = scanner.create_session(four_image_paths)

    assert len(session.views) == 4

    angles = [v.view_angle for v in session.views]
    assert angles == ["front", "right", "back", "left"]


# --------------------------------------------------------------------------- #
# Test 2: 4 images → coverage score = 1.0
# --------------------------------------------------------------------------- #

def test_coverage_score_four_images(four_image_paths: list[str]):
    """4 distinct angles cover all 4 sides → coverage_score == 1.0."""
    scanner = MultiAngleScanner()
    session = scanner.create_session(four_image_paths)

    assert session.coverage_score == 1.0
    assert session.has_front is True
    assert session.has_back is True
    assert session.has_left is True
    assert session.has_right is True


# --------------------------------------------------------------------------- #
# Test 3: 2 images → coverage score = 0.5
# --------------------------------------------------------------------------- #

def test_coverage_score_two_images(two_image_paths: list[str]):
    """2 images (front + back) cover 2 of 4 sides → coverage_score == 0.5."""
    scanner = MultiAngleScanner()
    session = scanner.create_session(two_image_paths)

    assert len(session.views) == 2
    # angles should be "front" and "back"
    angles = {v.view_angle for v in session.views}
    assert angles == {"front", "back"}
    # coverage: 2 of 4 sides
    assert session.coverage_score == 0.5


# --------------------------------------------------------------------------- #
# Test 4: create_reference_sheet generates an image file
# --------------------------------------------------------------------------- #

def test_create_reference_sheet_generates_image(four_image_paths: list[str]):
    """Reference sheet is saved as a valid JPEG with correct size."""
    scanner = MultiAngleScanner()
    session = scanner.create_session(four_image_paths)

    fd, output_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        result = scanner.create_reference_sheet(session, output_path, cols=2)

        assert result == output_path
        assert os.path.exists(output_path)
        assert os.path.getsize(output_path) > 0

        # Verify it's a valid image
        img = Image.open(output_path)
        assert img.format == "JPEG"
        assert img.mode == "RGB"
        assert img.size[0] > 0 and img.size[1] > 0
    finally:
        os.unlink(output_path)


# --------------------------------------------------------------------------- #
# Test 5: ViewImage dataclass fields are populated correctly
# --------------------------------------------------------------------------- #

def test_view_image_data_correct(four_image_paths: list[str]):
    """ViewImage stores path, angle, confidence, width, height accurately."""
    scanner = MultiAngleScanner()
    session = scanner.create_session(four_image_paths)

    for view in session.views:
        assert isinstance(view, ViewImage)
        assert view.path in four_image_paths        # path matches an input
        assert view.view_angle in {"front", "right", "back", "left"}
        assert 0.0 <= view.confidence <= 1.0
        assert view.width > 0
        assert view.height > 0

    # Check specific values for the first view (front)
    front_view = session.views[0]
    assert front_view.view_angle == "front"
    # width/height match the actual image
    img = Image.open(front_view.path)
    assert front_view.width == img.width
    assert front_view.height == img.height
