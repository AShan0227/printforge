"""Server-Sent Events (SSE) for real-time generation progress.

Usage:
    curl -N http://localhost:8000/api/v2/events?api_key=pf_xxx

Events:
    generation.started  {"task_id": "gen_xxx", "backend": "local"}
    generation.progress {"task_id": "gen_xxx", "stage": "inference", "percent": 50}
    generation.done     {"task_id": "gen_xxx", "vertices": 18000, "duration_ms": 10500}
    generation.failed   {"task_id": "gen_xxx", "error": "..."}
    queue.update        {"size": 3, "active": 1}
"""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator, Dict, Set

logger = logging.getLogger(__name__)


class EventBus:
    """Simple pub/sub event bus for SSE clients."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        return cls._instance

    def subscribe(self, channel: str = "default") -> asyncio.Queue:
        """Subscribe to events. Returns a queue to await."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        if channel not in self._subscribers:
            self._subscribers[channel] = set()
        self._subscribers[channel].add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue, channel: str = "default"):
        """Unsubscribe from events."""
        if channel in self._subscribers:
            self._subscribers[channel].discard(q)

    def publish(self, event_type: str, data: dict, channel: str = "default"):
        """Publish an event to all subscribers."""
        message = {"event": event_type, "data": data, "timestamp": time.time()}
        for channel_name in (channel, "default"):
            for q in self._subscribers.get(channel_name, set()).copy():
                try:
                    q.put_nowait(message)
                except asyncio.QueueFull:
                    pass  # Drop events for slow clients

    async def stream(self, channel: str = "default") -> AsyncGenerator[str, None]:
        """Yield SSE-formatted events."""
        q = self.subscribe(channel)
        try:
            while True:
                msg = await asyncio.wait_for(q.get(), timeout=30)
                yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"
        except asyncio.TimeoutError:
            # Send keepalive
            yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            self.unsubscribe(q, channel)


# Global event bus
event_bus = EventBus()


def emit_generation_started(task_id: str, backend: str, user_id: str = ""):
    event_bus.publish("generation.started", {
        "task_id": task_id, "backend": backend, "user_id": user_id,
    })


def emit_generation_progress(task_id: str, stage: str, percent: float):
    event_bus.publish("generation.progress", {
        "task_id": task_id, "stage": stage, "percent": round(percent * 100),
    })


def emit_generation_done(task_id: str, vertices: int, faces: int, duration_ms: int):
    event_bus.publish("generation.done", {
        "task_id": task_id, "vertices": vertices, "faces": faces, "duration_ms": duration_ms,
    })


def emit_generation_failed(task_id: str, error: str):
    event_bus.publish("generation.failed", {"task_id": task_id, "error": error})


def emit_queue_update(size: int, active: int):
    event_bus.publish("queue.update", {"size": size, "active": active})
