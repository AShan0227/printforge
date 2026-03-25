"""Content Safety: Image validation, rate limiting, and safety checks."""

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Limits
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB
MAX_IMAGE_DIMENSION = 8192  # 8K pixels
MIN_IMAGE_DIMENSION = 32
VALID_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}

# NSFW detection thresholds
SKIN_TONE_THRESHOLD = 0.60  # flag if >60% of pixels are skin-toned

# Rate limiting
DEFAULT_RATE_LIMIT = 20  # generations per hour per IP
DEFAULT_RATE_WINDOW = 3600  # 1 hour in seconds


@dataclass
class SafetyResult:
    """Result of content safety check."""
    safe: bool
    flags: List[str] = field(default_factory=list)

    @property
    def reason(self) -> Optional[str]:
        if self.flags:
            return "; ".join(self.flags)
        return None


def image_hash(image_path: str) -> str:
    """Compute SHA-256 hash of an image file for blocklist lookups.

    Args:
        image_path: Path to the image file.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class ContentSafety:
    """Content safety checks for uploaded images.

    Checks:
    - File size limit (50MB)
    - Image dimensions (min 32px, max 8192px)
    - Valid image format
    - NSFW heuristic: excessive skin-tone pixel ratio
    - Banned image hash blocklist
    """

    def __init__(self, banned_hashes: Optional[Set[str]] = None):
        self._banned_hashes: Set[str] = banned_hashes or set()

    def check_image(self, image_path: str) -> SafetyResult:
        """Run safety checks on an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            SafetyResult indicating whether the image is safe.
        """
        path = Path(image_path)
        flags: List[str] = []

        # Check file exists
        if not path.exists():
            return SafetyResult(safe=False, flags=["File not found"])

        # Check file size
        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            size_mb = file_size / (1024 * 1024)
            flags.append(f"File too large: {size_mb:.1f}MB (max 50MB)")

        if file_size == 0:
            flags.append("File is empty")

        # Check format
        suffix = path.suffix.lower()
        if suffix not in VALID_IMAGE_FORMATS:
            flags.append(f"Invalid format: {suffix} (allowed: {', '.join(sorted(VALID_IMAGE_FORMATS))})")

        # Check image dimensions (only if format is valid and no size issues)
        if not flags:
            dim_flags = self._check_dimensions(path)
            flags.extend(dim_flags)

        # Content safety: skin-tone heuristic + banned hash check
        if not flags:
            flags.extend(self._check_nsfw(path))

        # Banned hash check
        if not flags:
            h = image_hash(image_path)
            if h in self._banned_hashes:
                flags.append("Image matches banned content hash")

        safe = len(flags) == 0
        if not safe:
            logger.warning("Safety check failed for %s: %s", image_path, flags)

        return SafetyResult(safe=safe, flags=flags)

    def _check_dimensions(self, path: Path) -> List[str]:
        """Check image dimensions are within acceptable range."""
        flags = []
        try:
            from PIL import Image
            with Image.open(str(path)) as img:
                width, height = img.size

                if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                    flags.append(
                        f"Image too large: {width}x{height} (max {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
                    )
                if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                    flags.append(
                        f"Image too small: {width}x{height} (min {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION})"
                    )
        except Exception as e:
            flags.append(f"Cannot read image: {e}")
        return flags

    def _check_nsfw(self, path: Path) -> List[str]:
        """Heuristic NSFW check based on skin-tone pixel ratio.

        Uses HSV color space to detect skin-tone pixels.  If >60% of
        the image consists of skin-tone colors the image is flagged.
        This is a lightweight heuristic, not a full ML classifier.
        """
        flags: List[str] = []
        try:
            from PIL import Image
            import numpy as np

            with Image.open(str(path)) as img:
                rgb = img.convert("RGB").resize((128, 128))
                arr = np.array(rgb, dtype=np.float32)

            # Convert RGB → HSV manually (avoid opencv dependency)
            r, g, b = arr[:, :, 0] / 255.0, arr[:, :, 1] / 255.0, arr[:, :, 2] / 255.0
            cmax = np.maximum(np.maximum(r, g), b)
            cmin = np.minimum(np.minimum(r, g), b)
            delta = cmax - cmin

            # Hue (in degrees)
            hue = np.zeros_like(delta)
            mask = delta > 0
            # Where max is R
            rm = mask & (cmax == r)
            hue[rm] = 60.0 * (((g[rm] - b[rm]) / delta[rm]) % 6)
            # Where max is G
            gm = mask & (cmax == g)
            hue[gm] = 60.0 * (((b[gm] - r[gm]) / delta[gm]) + 2)
            # Where max is B
            bm = mask & (cmax == b)
            hue[bm] = 60.0 * (((r[bm] - g[bm]) / delta[bm]) + 4)

            sat = np.where(cmax > 0, delta / cmax, 0)
            val = cmax

            # Skin tone ranges in HSV:
            # Hue: 0-50° (reds/oranges/yellows)
            # Saturation: 0.15-0.75
            # Value: 0.2-0.95
            skin_mask = (
                ((hue >= 0) & (hue <= 50)) &
                (sat >= 0.15) & (sat <= 0.75) &
                (val >= 0.20) & (val <= 0.95)
            )

            skin_ratio = float(np.mean(skin_mask))
            if skin_ratio > SKIN_TONE_THRESHOLD:
                flags.append(
                    f"Image flagged: excessive skin-tone pixels ({skin_ratio:.0%} > {SKIN_TONE_THRESHOLD:.0%})"
                )
        except Exception as e:
            logger.warning("NSFW heuristic check failed: %s", e)
        return flags


class RateLimiter:
    """In-memory rate limiter for API endpoints.

    Tracks requests per IP with a sliding window.
    Thread-safe via Lock.
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_RATE_LIMIT,
        window_seconds: int = DEFAULT_RATE_WINDOW,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._hash_requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()

    def check(self, client_ip: str, img_hash: Optional[str] = None) -> Tuple[bool, int]:
        """Check if a request is allowed for the given IP (and optional image hash).

        When img_hash is provided, the same image counts toward a
        per-hash limit (same window, same max).  This prevents a
        single image from being submitted repeatedly.

        Args:
            client_ip: The client's IP address.
            img_hash: Optional SHA-256 hash of the uploaded image.

        Returns:
            Tuple of (allowed: bool, remaining: int).
        """
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            # Per-IP check
            self._requests[client_ip] = [
                t for t in self._requests[client_ip] if t > cutoff
            ]

            current_count = len(self._requests[client_ip])
            remaining = max(0, self.max_requests - current_count)

            if current_count >= self.max_requests:
                return False, 0

            # Per-hash check (same image submitted too many times)
            if img_hash:
                self._hash_requests[img_hash] = [
                    t for t in self._hash_requests[img_hash] if t > cutoff
                ]
                hash_count = len(self._hash_requests[img_hash])
                if hash_count >= self.max_requests:
                    return False, 0
                self._hash_requests[img_hash].append(now)

            self._requests[client_ip].append(now)
            return True, remaining - 1

    def get_usage(self, client_ip: str) -> Dict:
        """Get rate limit usage for a client IP."""
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            self._requests[client_ip] = [
                t for t in self._requests[client_ip] if t > cutoff
            ]
            count = len(self._requests[client_ip])

        return {
            "requests_used": count,
            "requests_remaining": max(0, self.max_requests - count),
            "limit": self.max_requests,
            "window_seconds": self.window_seconds,
        }


# ── Magic bytes validation ─────────────────────────────────────────

# Image format signatures (magic bytes)
_IMAGE_SIGNATURES = {
    "jpeg": [b"\xff\xd8\xff"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "bmp": [b"BM"],
    "webp": [b"RIFF"],  # RIFF header; full check includes WEBP at offset 8
    "gif": [b"GIF87a", b"GIF89a"],
    "tiff": [b"II\x2a\x00", b"MM\x00\x2a"],
}


def validate_image_magic_bytes(data: bytes) -> Tuple[bool, str]:
    """Validate that file content starts with known image magic bytes.

    Args:
        data: Raw file bytes.

    Returns:
        Tuple of (is_valid_image: bool, detected_type: str).
        detected_type is the format name if valid, or "unknown" if not.
    """
    if not data:
        return False, "empty"

    for fmt, signatures in _IMAGE_SIGNATURES.items():
        for sig in signatures:
            if data[:len(sig)] == sig:
                # Extra check for WEBP: bytes 8-12 must be "WEBP"
                if fmt == "webp" and data[8:12] != b"WEBP":
                    continue
                return True, fmt

    return False, "unknown"
