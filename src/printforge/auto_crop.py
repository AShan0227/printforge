"""Auto-crop: detect subject via edge detection + contour bounding box."""

from __future__ import annotations

import numpy as np
from PIL import Image

__all__ = ["auto_crop"]


def auto_crop(image: Image.Image, padding: float = 0.1) -> Image.Image:
    """Crop image to the detected subject region.

    Strategy:
      1. Convert to grayscale.
      2. Apply Gaussian blur to suppress noise.
      3. Sobel edge detection via scipy.ndimage.
      4. Threshold → binary edge map.
      5. Find largest connected component → bounding box.
      6. Expand bbox by `padding` fraction.
      7. If detection fails or bbox is tiny return original image.

    Args:
        image: PIL Image (any mode).
        padding: Fraction of width/height to add as margin (default 0.1 = 10%).

    Returns:
        Cropped PIL Image, or original if detection fails.
    """
    from scipy import ndimage

    # --- Grayscale ---
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)

    # --- Blur ---
    blurred = ndimage.gaussian_filter(arr, sigma=2.0)

    # --- Sobel edge detection ---
    sx = ndimage.sobel(blurred, axis=1, mode="reflect")
    sy = ndimage.sobel(blurred, axis=0, mode="reflect")
    magnitude = np.hypot(sx, sy)

    # --- Threshold ---
    threshold = magnitude.max() * 0.05
    binary = (magnitude > threshold).astype(np.uint8)

    # --- Dilate slightly to bridge small gaps in edges ---
    struct = ndimage.generate_binary_structure(2, 1)
    binary = ndimage.binary_dilation(binary, structure=struct, iterations=2).astype(np.uint8)

    # --- Connected components ---
    labeled, num_features = ndimage.label(binary)
    if num_features == 0:
        return image

    # Find the component with the largest area (most edge pixels)
    sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    max_idx = int(np.argmax(sizes)) + 1

    # Bounding box of the largest component
    slices = ndimage.find_objects(labeled == max_idx)[0]
    y0, y1 = slices[0].start, slices[0].stop
    x0, x1 = slices[1].start, slices[1].stop

    w, h = x1 - x0, y1 - y0

    # Guard: if bbox is tiny relative to image, bail
    if w < 20 or h < 20:
        return image

    # --- Apply padding ---
    img_w, img_h = image.size
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(img_w, x1 + pad_x)
    y1 = min(img_h, y1 + pad_y)

    return image.crop((x0, y0, x1, y1))
