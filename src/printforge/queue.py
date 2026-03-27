"""Async generation queue for PrintForge v2.2.

Supports submitting generation jobs that run in the background,
querying status by task_id, and retrieving results when done.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class GenerationTask:
    task_id: str
    status: TaskStatus
    created_at: str
    image_path: str
    output_format: str = "stl"
    backend: str = "auto"
    multi_view: bool = False
    scale_mm: float = 50.0
    add_base: bool = False
    user_id: str = "anonymous"
    api_key: str = ""
    # Results (filled after completion)
    result_path: Optional[str] = None
    vertices: int = 0
    faces: int = 0
    is_watertight: bool = False
    duration_ms: int = 0
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output_format": self.output_format,
            "backend": self.backend,
            "multi_view": self.multi_view,
            "result_path": self.result_path,
            "vertices": self.vertices,
            "faces": self.faces,
            "is_watertight": self.is_watertight,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


class GenerationQueue:
    """Singleton async generation queue."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._tasks: Dict[str, GenerationTask] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers_started = False
        self._max_workers = 2
        self._initialized = True

    async def ensure_workers(self):
        """Start background workers if not already running."""
        if self._workers_started:
            return
        self._workers_started = True
        for i in range(self._max_workers):
            asyncio.create_task(self._worker(i))
            logger.info(f"Generation worker {i} started")

    def submit(
        self,
        image_path: str,
        output_format: str = "stl",
        backend: str = "auto",
        multi_view: bool = False,
        scale_mm: float = 50.0,
        add_base: bool = False,
        user_id: str = "anonymous",
        api_key: str = "",
    ) -> str:
        """Submit a generation task. Returns task_id."""
        task_id = f"gen_{uuid.uuid4().hex[:12]}"
        task = GenerationTask(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc).isoformat(),
            image_path=image_path,
            output_format=output_format,
            backend=backend,
            multi_view=multi_view,
            scale_mm=scale_mm,
            add_base=add_base,
            user_id=user_id,
            api_key=api_key,
        )
        self._tasks[task_id] = task
        self._queue.put_nowait(task)
        logger.info(f"Task {task_id} queued ({self._queue.qsize()} in queue)")
        return task_id

    def get_status(self, task_id: str) -> Optional[Dict]:
        """Get task status."""
        task = self._tasks.get(task_id)
        if not task:
            return None
        return task.to_dict()

    def get_task(self, task_id: str) -> Optional[GenerationTask]:
        """Get full task object."""
        return self._tasks.get(task_id)

    def list_tasks(self, user_id: Optional[str] = None, limit: int = 20) -> list:
        """List recent tasks."""
        tasks = list(self._tasks.values())
        if user_id:
            tasks = [t for t in tasks if t.user_id == user_id]
        return [t.to_dict() for t in sorted(tasks, key=lambda t: t.created_at, reverse=True)[:limit]]

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def active_count(self) -> int:
        return sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)

    async def _worker(self, worker_id: int):
        """Background worker that processes generation tasks."""
        while True:
            task = await self._queue.get()
            try:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now(timezone.utc).isoformat()
                logger.info(f"Worker {worker_id}: processing {task.task_id}")

                await self._run_pipeline(task)

                task.status = TaskStatus.DONE
                task.finished_at = datetime.now(timezone.utc).isoformat()
                logger.info(f"Worker {worker_id}: completed {task.task_id} in {task.duration_ms}ms")
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.finished_at = datetime.now(timezone.utc).isoformat()
                logger.error(f"Worker {worker_id}: failed {task.task_id}: {e}")
            finally:
                self._queue.task_done()

    async def _run_pipeline(self, task: GenerationTask):
        """Run the PrintForge pipeline for a task."""
        import tempfile
        from .pipeline import PrintForgePipeline, PipelineConfig

        output_suffix = ".3mf" if task.output_format == "3mf" else f".{task.output_format}"
        with tempfile.NamedTemporaryFile(suffix=output_suffix, delete=False) as tmp:
            output_path = tmp.name

        config = PipelineConfig(
            inference_backend=task.backend,
            output_format=task.output_format,
            scale_mm=task.scale_mm,
            add_base=task.add_base,
            multi_view=task.multi_view,
        )

        pipeline = PrintForgePipeline(config)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: pipeline.run(task.image_path, output_path)
        )

        task.result_path = output_path
        task.vertices = result.vertices
        task.faces = result.faces
        task.is_watertight = result.is_watertight
        task.duration_ms = int(result.duration_ms)
