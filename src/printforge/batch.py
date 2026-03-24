"""Batch Processing: Process multiple images to 3D models in parallel."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .pipeline import PrintForgePipeline, PipelineConfig, PipelineResult

logger = logging.getLogger(__name__)


@dataclass
class BatchItem:
    """Result for a single item in a batch."""
    input_path: str
    output_path: Optional[str] = None
    result: Optional[PipelineResult] = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.result is not None and self.error is None


@dataclass
class BatchResult:
    """Aggregate result of a batch processing run."""
    items: List[BatchItem] = field(default_factory=list)
    total_duration_ms: float = 0.0

    @property
    def succeeded(self) -> int:
        return sum(1 for item in self.items if item.success)

    @property
    def failed(self) -> int:
        return sum(1 for item in self.items if not item.success)


class BatchProcessor:
    """Process multiple images to 3D models in parallel.

    Args:
        config: Pipeline configuration shared by all items.
        max_workers: Maximum number of parallel pipeline runs (default: 3).
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        max_workers: int = 3,
    ):
        self.config = config or PipelineConfig()
        self.max_workers = max_workers

    def process(
        self,
        image_paths: List[str],
        output_dir: str,
        output_format: str = "3mf",
        progress_callback: Optional[Callable[[int, int, BatchItem], None]] = None,
    ) -> BatchResult:
        """Process a list of images into 3D models.

        Args:
            image_paths: List of input image file paths.
            output_dir: Directory where output files are written.
            output_format: Output format ("3mf", "stl", "obj").
            progress_callback: Optional callable(completed_count, total, item)
                               called after each item finishes.

        Returns:
            BatchResult with per-item results.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        batch_start = time.time()
        total = len(image_paths)
        items: List[BatchItem] = []
        completed = 0

        def _run_one(image_path: str) -> BatchItem:
            stem = Path(image_path).stem
            output_path = str(out / f"{stem}.{output_format}")

            item_start = time.time()
            try:
                cfg = PipelineConfig(
                    device=self.config.device,
                    inference_backend=self.config.inference_backend,
                    mc_resolution=self.config.mc_resolution,
                    scale_mm=self.config.scale_mm,
                    max_faces=self.config.max_faces,
                    add_base=self.config.add_base,
                    output_format=output_format,
                )
                pipeline = PrintForgePipeline(cfg)
                result = pipeline.run(image_path, output_path)
                return BatchItem(
                    input_path=image_path,
                    output_path=output_path,
                    result=result,
                    duration_ms=(time.time() - item_start) * 1000,
                )
            except Exception as e:
                logger.error("Batch item failed: %s — %s", image_path, e)
                return BatchItem(
                    input_path=image_path,
                    error=str(e),
                    duration_ms=(time.time() - item_start) * 1000,
                )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_run_one, path): path
                for path in image_paths
            }
            for future in as_completed(futures):
                item = future.result()
                items.append(item)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, item)

        return BatchResult(
            items=items,
            total_duration_ms=(time.time() - batch_start) * 1000,
        )

    @classmethod
    def collect_images(cls, directory: str) -> List[str]:
        """Collect all image files from a directory (non-recursive)."""
        d = Path(directory)
        if not d.is_dir():
            raise FileNotFoundError(f"Not a directory: {directory}")

        paths = sorted(
            str(p) for p in d.iterdir()
            if p.is_file() and p.suffix.lower() in cls.IMAGE_EXTENSIONS
        )
        return paths
