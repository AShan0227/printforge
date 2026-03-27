"""Health check module for PrintForge.

Checks:
- TripoSR model loaded status
- Disk space availability
- Memory usage
"""

import os
try:
    import psutil
except ImportError:
    psutil = None  # type: ignore
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_DISK_SPACE_GB = 5.0
MAX_MEMORY_PERCENT = 90.0


@dataclass
class HealthCheckResult:
    name: str
    healthy: bool
    value: Any = None
    message: str = ""
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class HealthChecker:
    """Performs comprehensive health checks on PrintForge components."""

    def __init__(self):
        self._model_loaded = False
        self._model_path: str = ""

    def register_model_status(self, loaded: bool, model_path: str = "") -> None:
        """Register the TripoSR model loading status."""
        self._model_loaded = loaded
        self._model_path = model_path
        logger.info(f"Health checker: model status updated - loaded={loaded}, path={model_path}")

    def check_model_loaded(self) -> HealthCheckResult:
        """Check if TripoSR model is loaded and ready."""
        status = self._model_loaded
        return HealthCheckResult(
            name="model_loaded",
            healthy=status,
            value=status,
            message="TripoSR model loaded" if status else "TripoSR model not loaded",
            details={"model_path": self._model_path} if self._model_path else {},
        )

    def check_disk_space(self, path: str = "/") -> HealthCheckResult:
        """Check available disk space."""
        try:
            stat = os.statvfs(path)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
            total_gb = (stat.f_blocks * stat.f_frsize) / (1024 ** 3)
            used_percent = ((stat.f_blocks - stat.f_bavail) / stat.f_blocks * 100) if stat.f_blocks > 0 else 0

            healthy = free_gb >= MIN_DISK_SPACE_GB
            return HealthCheckResult(
                name="disk_space",
                healthy=healthy,
                value=round(free_gb, 2),
                message=f"{free_gb:.1f} GB free" if healthy else f"Low disk space: {free_gb:.1f} GB free",
                details={
                    "free_gb": round(free_gb, 2),
                    "total_gb": round(total_gb, 2),
                    "used_percent": round(used_percent, 1),
                    "min_required_gb": MIN_DISK_SPACE_GB,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to check disk space: {e}")
            return HealthCheckResult(
                name="disk_space",
                healthy=False,
                value=None,
                message=f"Failed to check disk space: {e}",
            )

    def check_memory(self) -> HealthCheckResult:
        """Check system memory usage."""
        try:
            if psutil is None:
                return HealthCheckResult(name="memory", healthy=True, value=0, message="psutil not installed")
            mem = psutil.virtual_memory()
            used_gb = mem.used / (1024 ** 3)
            total_gb = mem.total / (1024 ** 3)
            percent = mem.percent

            healthy = percent < MAX_MEMORY_PERCENT
            return HealthCheckResult(
                name="memory",
                healthy=healthy,
                value=round(percent, 1),
                message=f"{percent:.1f}% used ({used_gb:.1f}/{total_gb:.1f} GB)",
                details={
                    "used_gb": round(used_gb, 1),
                    "total_gb": round(total_gb, 1),
                    "percent": round(percent, 1),
                    "max_percent": MAX_MEMORY_PERCENT,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to check memory: {e}")
            return HealthCheckResult(
                name="memory",
                healthy=False,
                value=None,
                message=f"Failed to check memory: {e}",
            )

    def check_all(self) -> Dict[str, Any]:
        """Run all health checks and return comprehensive status."""
        checks: List[HealthCheckResult] = [
            self.check_model_loaded(),
            self.check_disk_space(),
            self.check_memory(),
        ]

        all_healthy = all(c.healthy for c in checks)
        healthy_checks = [c.name for c in checks if c.healthy]
        unhealthy_checks = [c.name for c in checks if not c.healthy]

        return {
            "status": "ok" if all_healthy else "degraded",
            "version": "1.4.0",
            "healthy": all_healthy,
            "checks": {c.name: {"healthy": c.healthy, "value": c.value, "message": c.message, "details": c.details}
                        for c in checks},
            "healthy_checks": healthy_checks,
            "unhealthy_checks": unhealthy_checks if unhealthy_checks else None,
        }


_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get the global health checker instance."""
    return _health_checker


def register_model_loaded(loaded: bool = True, model_path: str = "") -> None:
    """Convenience function to register model status."""
    _health_checker.register_model_status(loaded, model_path)
