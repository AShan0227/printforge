"""PrintForge Image Cache — SHA256-based mesh caching layer."""

import hashlib
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".printforge", "cache")
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


@dataclass
class CacheStats:
    """Statistics about the image cache."""
    hits: int
    misses: int
    size_bytes: int
    num_entries: int


class ImageCache:
    """SHA256-based cache for image-to-mesh results.

    Stores cached mesh files keyed by the SHA256 hash of the input image bytes.
    Thread-safe via a reentrant lock.
    """

    def __init__(
        self,
        cache_dir: str = DEFAULT_CACHE_DIR,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_bytes(data: bytes) -> str:
        """Compute SHA256 hex digest of raw bytes."""
        return hashlib.sha256(data).hexdigest()

    def _entry_dir(self, key: str) -> Path:
        """Return the cache subdirectory for a given hash key."""
        # Use first 2 chars as bucket to avoid huge flat directories
        return self.cache_dir / key[:2] / key

    def _is_expired(self, entry_dir: Path) -> bool:
        """Check whether a cache entry has exceeded its TTL."""
        meta_file = entry_dir / ".timestamp"
        ttl_file = entry_dir / ".ttl"
        if not meta_file.exists():
            return True
        try:
            ts = float(meta_file.read_text().strip())
            ttl = self.ttl_seconds
            if ttl_file.exists():
                ttl = float(ttl_file.read_text().strip())
            return (time.time() - ts) > ttl
        except (ValueError, OSError):
            return True

    @staticmethod
    def _ensure_bytes(data) -> bytes:
        """Ensure data is bytes — encode strings automatically."""
        if isinstance(data, str):
            return data.encode("utf-8")
        return data

    def get(self, image_bytes) -> Optional[str]:
        """Look up a cached mesh path for the given image bytes.

        Returns:
            Absolute path to the cached mesh file, or None on miss.
        """
        key = self._hash_bytes(self._ensure_bytes(image_bytes))
        entry_dir = self._entry_dir(key)

        with self._lock:
            if not entry_dir.exists():
                self._misses += 1
                return None

            if self._is_expired(entry_dir):
                logger.debug("Cache entry expired for %s", key[:12])
                shutil.rmtree(entry_dir, ignore_errors=True)
                self._misses += 1
                return None

            # Find the mesh file inside the entry directory
            mesh_files = [
                f for f in entry_dir.iterdir()
                if f.suffix in (".stl", ".3mf", ".obj", ".glb") and f.is_file()
            ]
            if not mesh_files:
                self._misses += 1
                return None

            self._hits += 1
            cached_path = str(mesh_files[0])
            logger.info("Cache hit for %s -> %s", key[:12], cached_path)
            return cached_path

    def put(self, image_bytes, mesh_path: str, ttl_seconds: Optional[int] = None) -> str:
        """Copy a mesh file into the cache, keyed by image SHA256.

        Args:
            image_bytes: Raw bytes of the input image (or string key).
            mesh_path: Path to the mesh file to cache.
            ttl_seconds: Per-entry TTL override. If 0, the entry is immediately expired.

        Returns:
            Absolute path to the cached copy of the mesh file.
        """
        key = self._hash_bytes(self._ensure_bytes(image_bytes))
        entry_dir = self._entry_dir(key)

        with self._lock:
            entry_dir.mkdir(parents=True, exist_ok=True)

            # Write timestamp and optional per-entry TTL
            (entry_dir / ".timestamp").write_text(str(time.time()))
            if ttl_seconds is not None:
                (entry_dir / ".ttl").write_text(str(ttl_seconds))

            # Copy mesh into cache
            src = Path(mesh_path)
            dest = entry_dir / src.name
            shutil.copy2(str(src), str(dest))

            logger.info("Cached mesh %s -> %s", key[:12], dest)
            return str(dest)

    def stats(self) -> CacheStats:
        """Return current cache statistics."""
        with self._lock:
            total_size = 0
            num_entries = 0

            if self.cache_dir.exists():
                for bucket in self.cache_dir.iterdir():
                    if not bucket.is_dir():
                        continue
                    for entry in bucket.iterdir():
                        if not entry.is_dir():
                            continue
                        num_entries += 1
                        for f in entry.iterdir():
                            if f.is_file():
                                total_size += f.stat().st_size

            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                size_bytes=total_size,
                num_entries=num_entries,
            )

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            if self.cache_dir.exists():
                shutil.rmtree(str(self.cache_dir), ignore_errors=True)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._hits = 0
            self._misses = 0
            logger.info("Cache cleared")
