"""Content Safety: Image validation, rate limiting, and safety checks."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Limits
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50MB
MAX_IMAGE_DIMENSION = 8192  # 8K pixels
MIN_IMAGE_DIMENSION = 32
VALID_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}

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


class ContentSafety:
    """Content safety checks for uploaded images.

    Checks:
    - File size limit (50MB)
    - Image dimensions (min 32px, max 8192px)
    - Valid image format
    - Placeholder for NSFW detection (returns safe=True with TODO)
    """

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

        # TODO: NSFW detection placeholder
        # When implemented, this would use a classification model to detect
        # inappropriate content. For now, we pass all content checks.
        # nsfw_flags = self._check_nsfw(path)
        # flags.extend(nsfw_flags)

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
        self._lock = Lock()

    def check(self, client_ip: str) -> Tuple[bool, int]:
        """Check if a request is allowed for the given IP.

        Args:
            client_ip: The client's IP address.

        Returns:
            Tuple of (allowed: bool, remaining: int).
        """
        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            # Remove expired timestamps
            self._requests[client_ip] = [
                t for t in self._requests[client_ip] if t > cutoff
            ]

            current_count = len(self._requests[client_ip])
            remaining = max(0, self.max_requests - current_count)

            if current_count >= self.max_requests:
                return False, 0

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
