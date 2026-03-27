"""Prometheus-format metrics collector for PrintForge."""

import time
import threading
from typing import Dict
from dataclasses import dataclass, field


@dataclass
class Counter:
    """Thread-safe counter metric."""
    value: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self.value += n

    def get(self) -> int:
        with self._lock:
            return self.value


@dataclass
class Gauge:
    """Thread-safe gauge metric."""
    value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, v: float) -> None:
        with self._lock:
            self.value = v

    def inc(self, n: float = 1.0) -> None:
        with self._lock:
            self.value += n

    def dec(self, n: float = 1.0) -> None:
        with self._lock:
            self.value -= n

    def get(self) -> float:
        with self._lock:
            return self.value


@dataclass
class Histogram:
    """Simple histogram for duration tracking."""
    _sum: float = 0.0
    _count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._count += 1

    def get(self) -> tuple[float, int]:
        with self._lock:
            return self._sum, self._count


class MetricsCollector:
    """Prometheus-format metrics collector.

    Tracks:
    - generation_count: Total number of successful generations
    - generation_duration_seconds: Histogram of generation durations
    - generation_errors_total: Total number of generation errors
    - active_generations: Number of currently running generations
    """

    def __init__(self) -> None:
        self.generation_count = Counter()
        self.generation_duration_seconds = Histogram()
        self.generation_errors_total = Counter()
        self.active_generations = Gauge()
        self._labels: Dict[str, Counter] = {}

    def record_generation_start(self) -> None:
        """Mark a generation as started (increment active counter)."""
        self.active_generations.inc()

    def record_generation_success(self, duration_seconds: float, backend: str = "unknown") -> None:
        """Record a successful generation completion."""
        self.generation_count.inc()
        self.generation_duration_seconds.observe(duration_seconds)
        self.active_generations.dec()
        self._inc_label("generation_backend", backend)

    def record_generation_error(self, error_type: str = "unknown") -> None:
        """Record a failed generation."""
        self.generation_errors_total.inc()
        self.active_generations.dec()
        self._inc_label("error_type", error_type)

    def _inc_label(self, label_type: str, label_value: str) -> None:
        """Increment a labeled counter."""
        key = f"{label_type}:{label_value}"
        if key not in self._labels:
            self._labels[key] = Counter()
        self._labels[key].inc()

    def expose_generation_start(self) -> float:
        """Return timestamp when a generation started (for duration calc)."""
        return time.time()

    def get_metrics_text(self) -> str:
        """Return Prometheus text format metrics."""
        lines = []

        lines.append("# HELP generation_count_total Total number of successful 3D generations")
        lines.append("# TYPE generation_count_total counter")
        lines.append(f"generation_count_total {self.generation_count.get()}")

        dur_sum, dur_count = self.generation_duration_seconds.get()
        lines.append("# HELP generation_duration_seconds Time spent generating 3D models")
        lines.append("# TYPE generation_duration_seconds summary")
        lines.append(f"generation_duration_seconds_sum {dur_sum:.6f}")
        lines.append(f"generation_duration_seconds_count {dur_count}")

        lines.append("# HELP generation_errors_total Total number of failed generations")
        lines.append("# TYPE generation_errors_total counter")
        lines.append(f"generation_errors_total {self.generation_errors_total.get()}")

        lines.append("# HELP active_generations Number of currently running generations")
        lines.append("# TYPE active_generations gauge")
        lines.append(f"active_generations {self.active_generations.get():.0f}")

        if self._labels:
            for key, counter in self._labels.items():
                label_type, label_value = key.split(":", 1)
                metric_name = f"generation_by_{label_type}"
                lines.append(f"# HELP {metric_name} Generations grouped by {label_type}")
                lines.append(f"# TYPE {metric_name} counter")
                lines.append(f'{metric_name}{{type="{label_value}"}} {counter.get()}')

        return "\n".join(lines) + "\n"


_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics
