"""Local Analytics: Track usage telemetry in a local SQLite database. No PII, no network calls."""

import logging
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".printforge" / "analytics.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    format TEXT,
    inference_backend TEXT,
    duration_ms REAL,
    quality_score REAL,
    metadata TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);
"""


class Analytics:
    """Purely local analytics stored in SQLite. No PII, no network calls.

    Tracks: generations, formats, inference_backend, duration, quality_score.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        """Yield a SQLite connection with auto-commit."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def track(
        self,
        event_type: str,
        format: Optional[str] = None,
        inference_backend: Optional[str] = None,
        duration_ms: Optional[float] = None,
        quality_score: Optional[float] = None,
        metadata: Optional[str] = None,
    ):
        """Record an analytics event.

        Args:
            event_type: Type of event (e.g. "generation", "text_to_3d", "optimize").
            format: Output format (e.g. "3mf", "stl").
            inference_backend: Backend used (e.g. "triposr", "hunyuan3d", "placeholder").
            duration_ms: Operation duration in milliseconds.
            quality_score: Quality score if applicable.
            metadata: Optional JSON string for additional data.
        """
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO events (event_type, format, inference_backend,
                   duration_ms, quality_score, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (event_type, format, inference_backend, duration_ms,
                 quality_score, metadata, time.time()),
            )
        logger.debug("Tracked event: %s", event_type)

    def get_stats(self) -> Dict[str, Any]:
        """Return a summary of all tracked analytics.

        Returns:
            Dict with keys: total_events, generations, formats, backends,
            avg_duration_ms, avg_quality_score, events_by_type, recent_events.
        """
        with self._conn() as conn:
            # Total events
            total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

            # Generations count
            generations = conn.execute(
                "SELECT COUNT(*) FROM events WHERE event_type = 'generation'"
            ).fetchone()[0]

            # Format breakdown
            format_rows = conn.execute(
                "SELECT format, COUNT(*) as cnt FROM events "
                "WHERE format IS NOT NULL GROUP BY format ORDER BY cnt DESC"
            ).fetchall()
            formats = {row["format"]: row["cnt"] for row in format_rows}

            # Backend breakdown
            backend_rows = conn.execute(
                "SELECT inference_backend, COUNT(*) as cnt FROM events "
                "WHERE inference_backend IS NOT NULL GROUP BY inference_backend ORDER BY cnt DESC"
            ).fetchall()
            backends = {row["inference_backend"]: row["cnt"] for row in backend_rows}

            # Average duration
            avg_dur = conn.execute(
                "SELECT AVG(duration_ms) FROM events WHERE duration_ms IS NOT NULL"
            ).fetchone()[0]

            # Average quality
            avg_qual = conn.execute(
                "SELECT AVG(quality_score) FROM events WHERE quality_score IS NOT NULL"
            ).fetchone()[0]

            # Events by type
            type_rows = conn.execute(
                "SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type ORDER BY cnt DESC"
            ).fetchall()
            events_by_type = {row["event_type"]: row["cnt"] for row in type_rows}

            # Recent events (last 10)
            recent_rows = conn.execute(
                "SELECT event_type, format, inference_backend, duration_ms, quality_score, created_at "
                "FROM events ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
            recent = [
                {
                    "event_type": row["event_type"],
                    "format": row["format"],
                    "inference_backend": row["inference_backend"],
                    "duration_ms": row["duration_ms"],
                    "quality_score": row["quality_score"],
                    "created_at": row["created_at"],
                }
                for row in recent_rows
            ]

        return {
            "total_events": total,
            "generations": generations,
            "formats": formats,
            "backends": backends,
            "avg_duration_ms": round(avg_dur, 1) if avg_dur else None,
            "avg_quality_score": round(avg_qual, 1) if avg_qual else None,
            "events_by_type": events_by_type,
            "recent_events": recent,
        }

    def clear(self):
        """Clear all analytics data."""
        with self._conn() as conn:
            conn.execute("DELETE FROM events")
        logger.info("Analytics data cleared")
