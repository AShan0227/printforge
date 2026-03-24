"""Performance Benchmarking Suite for PrintForge pipeline stages."""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of benchmarking a single inference backend."""
    backend: str
    duration_ms: float
    vertices: int = 0
    faces: int = 0
    quality_score: float = 0.0
    error: Optional[str] = None


@dataclass
class WatertightResult:
    """Result of benchmarking the watertight conversion."""
    duration_ms: float
    faces_before: int
    faces_after: int
    is_watertight: bool


@dataclass
class PipelineStageResult:
    """Timing for a single pipeline stage."""
    stage: str
    duration_ms: float


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    timestamp: str = ""
    image_path: str = ""
    inference_results: List[InferenceResult] = field(default_factory=list)
    watertight_result: Optional[WatertightResult] = None
    pipeline_stages: List[PipelineStageResult] = field(default_factory=list)
    total_pipeline_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "inference": [asdict(r) for r in self.inference_results],
            "watertight": asdict(self.watertight_result) if self.watertight_result else None,
            "pipeline_stages": [asdict(s) for s in self.pipeline_stages],
            "total_pipeline_ms": self.total_pipeline_ms,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# Path for storing the latest benchmark report
BENCHMARK_REPORT_PATH = Path.home() / ".printforge" / "benchmark_latest.json"


class BenchmarkSuite:
    """Run performance benchmarks on PrintForge pipeline components."""

    def benchmark_inference(
        self, image_path: str, backends: Optional[List[str]] = None
    ) -> List[InferenceResult]:
        """Benchmark inference across specified backends.

        Args:
            image_path: Path to test image.
            backends: List of backends to test. Defaults to ['placeholder', 'hunyuan3d'].

        Returns:
            List of InferenceResult with timing and quality for each backend.
        """
        from .pipeline import PrintForgePipeline, PipelineConfig
        from .quality import QualityScorer

        if backends is None:
            backends = ["placeholder", "hunyuan3d"]

        scorer = QualityScorer()
        results = []

        for backend in backends:
            config = PipelineConfig(
                inference_backend=backend,
                device="cpu",
            )
            pipeline = PrintForgePipeline(config)

            try:
                t0 = time.time()
                image = pipeline._load_image(image_path)
                raw_mesh = pipeline._infer_3d(image)
                duration_ms = (time.time() - t0) * 1000

                quality_score = 0.0
                try:
                    report = scorer.score(raw_mesh)
                    quality_score = report.total_score
                except Exception:
                    pass

                results.append(InferenceResult(
                    backend=backend,
                    duration_ms=round(duration_ms, 1),
                    vertices=len(raw_mesh.vertices),
                    faces=len(raw_mesh.faces),
                    quality_score=round(quality_score, 1),
                ))
            except Exception as e:
                results.append(InferenceResult(
                    backend=backend,
                    duration_ms=0.0,
                    error=str(e),
                ))

        return results

    def benchmark_watertight(self, mesh) -> WatertightResult:
        """Benchmark watertight conversion on a mesh.

        Args:
            mesh: A trimesh.Trimesh object.

        Returns:
            WatertightResult with timing and face counts.
        """
        from .pipeline import PrintForgePipeline, PipelineConfig

        faces_before = len(mesh.faces)
        pipeline = PrintForgePipeline(PipelineConfig(device="cpu"))

        t0 = time.time()
        watertight = pipeline._make_watertight(mesh)
        duration_ms = (time.time() - t0) * 1000

        return WatertightResult(
            duration_ms=round(duration_ms, 1),
            faces_before=faces_before,
            faces_after=len(watertight.faces),
            is_watertight=bool(watertight.is_watertight),
        )

    def benchmark_pipeline(self, image_path: str) -> BenchmarkReport:
        """Run a full pipeline benchmark with per-stage timing.

        Args:
            image_path: Path to test image.

        Returns:
            BenchmarkReport with detailed timing breakdown.
        """
        import tempfile
        from datetime import datetime, timezone

        from .pipeline import PrintForgePipeline, PipelineConfig

        config = PipelineConfig(
            inference_backend="placeholder",
            device="cpu",
            output_format="stl",
        )
        pipeline = PrintForgePipeline(config)

        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
            output_path = tmp.name

        t0 = time.time()
        result = pipeline.run(image_path, output_path)
        total_ms = (time.time() - t0) * 1000

        stages = [
            PipelineStageResult(stage=name, duration_ms=round(dur * 1000, 1))
            for name, dur in result.stages.items()
        ]

        report = BenchmarkReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            image_path=image_path,
            pipeline_stages=stages,
            total_pipeline_ms=round(total_ms, 1),
        )

        # Persist latest report
        self._save_report(report)

        # Clean up temp file
        try:
            Path(output_path).unlink(missing_ok=True)
        except Exception:
            pass

        return report

    def _save_report(self, report: BenchmarkReport):
        """Save the report to ~/.printforge/benchmark_latest.json."""
        BENCHMARK_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        BENCHMARK_REPORT_PATH.write_text(report.to_json())

    @staticmethod
    def load_latest() -> Optional[BenchmarkReport]:
        """Load the most recently saved benchmark report."""
        if not BENCHMARK_REPORT_PATH.exists():
            return None
        try:
            data = json.loads(BENCHMARK_REPORT_PATH.read_text())
            report = BenchmarkReport(
                timestamp=data.get("timestamp", ""),
                image_path=data.get("image_path", ""),
                total_pipeline_ms=data.get("total_pipeline_ms", 0.0),
            )
            for s in data.get("pipeline_stages", []):
                report.pipeline_stages.append(
                    PipelineStageResult(stage=s["stage"], duration_ms=s["duration_ms"])
                )
            for r in data.get("inference", []):
                report.inference_results.append(InferenceResult(
                    backend=r["backend"],
                    duration_ms=r["duration_ms"],
                    vertices=r.get("vertices", 0),
                    faces=r.get("faces", 0),
                    quality_score=r.get("quality_score", 0.0),
                    error=r.get("error"),
                ))
            wt = data.get("watertight")
            if wt:
                report.watertight_result = WatertightResult(**wt)
            return report
        except Exception as e:
            logger.warning("Failed to load benchmark report: %s", e)
            return None
