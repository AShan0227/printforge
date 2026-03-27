"""Tests for async task queue (queue.py) and /api/v2/generate/async endpoints."""

import asyncio
import tempfile
import time
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def sample_image_bytes():
    """Create a small valid JPEG image in memory."""
    img = Image.new("RGB", (128, 128), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestTaskQueue:
    def test_submit_returns_task_id(self, sample_image_bytes):
        from printforge.queue import TaskQueue, TaskInput

        queue = TaskQueue(max_workers=1, result_dir=Path(tempfile.mkdtemp()))
        task_input = TaskInput(
            image_bytes=sample_image_bytes,
            filename="test.jpg",
            format="stl",
            size_mm=30.0,
        )

        task_id = queue.submit(task_input)
        assert task_id.startswith("task_")
        assert len(task_id) == 17  # "task_" + 12 hex chars

    def test_get_status_pending(self, sample_image_bytes):
        from printforge.queue import TaskQueue, TaskInput, TaskStatus

        queue = TaskQueue(max_workers=1, result_dir=Path(tempfile.mkdtemp()))
        task_input = TaskInput(image_bytes=sample_image_bytes, filename="test.jpg")
        task_id = queue.submit(task_input)

        info = queue.get_status(task_id)
        assert info is not None
        assert info.task_id == task_id
        assert info.status == TaskStatus.PENDING
        assert info.result_path is None
        assert info.error is None

    def test_get_status_not_found(self):
        from printforge.queue import TaskQueue

        queue = TaskQueue()
        assert queue.get_status("task_nonexistent") is None

    def test_cancel_pending_task(self, sample_image_bytes):
        from printforge.queue import TaskQueue, TaskInput

        queue = TaskQueue(max_workers=1, result_dir=Path(tempfile.mkdtemp()))
        task_input = TaskInput(image_bytes=sample_image_bytes, filename="test.jpg")
        task_id = queue.submit(task_input)

        assert queue.cancel(task_id) is True
        info = queue.get_status(task_id)
        assert info.status.value == "failed"
        assert "Cancelled" in info.error

    def test_cancel_running_task_fails(self, sample_image_bytes):
        from printforge.queue import TaskQueue, TaskInput

        queue = TaskQueue(max_workers=1, result_dir=Path(tempfile.mkdtemp()))
        task_input = TaskInput(
            image_bytes=sample_image_bytes,
            filename="test.jpg",
            backend="placeholder",  # fast backend
        )
        task_id = queue.submit(task_input)

        # Wait a tiny bit for it to start
        time.sleep(0.2)
        assert queue.cancel(task_id) is False  # already running

    @pytest.mark.asyncio
    async def test_async_submit(self, sample_image_bytes):
        from printforge.queue import TaskQueue, TaskInput

        queue = TaskQueue(max_workers=1, result_dir=Path(tempfile.mkdtemp()))
        task_input = TaskInput(image_bytes=sample_image_bytes, filename="async_test.jpg")

        task_id = await queue.submit_async(task_input)
        assert task_id.startswith("task_")

        info = queue.get_status(task_id)
        assert info.status.value == "pending"

    def test_list_tasks(self, sample_image_bytes):
        from printforge.queue import TaskQueue, TaskInput

        queue = TaskQueue(max_workers=1, result_dir=Path(tempfile.mkdtemp()))
        task_ids = []
        for i in range(3):
            inp = TaskInput(image_bytes=sample_image_bytes, filename=f"img{i}.jpg")
            tid = queue.submit(inp)
            task_ids.append(tid)

        tasks = queue.list_tasks(limit=10)
        assert len(tasks) == 3
        # Most recent first
        assert tasks[0].task_id == task_ids[-1]

    def test_task_info_duration(self, sample_image_bytes):
        from printforge.queue import TaskQueue, TaskInput

        queue = TaskQueue(max_workers=1, result_dir=Path(tempfile.mkdtemp()))
        task_input = TaskInput(image_bytes=sample_image_bytes, filename="test.jpg")
        task_id = queue.submit(task_input)

        info = queue.get_status(task_id)
        assert info.duration_ms() is None  # not started

        # Simulate started state
        info.started_at = time.time() - 5.0
        info.completed_at = time.time()
        assert info.duration_ms() == pytest.approx(5000, rel=100)
        assert info.is_terminal is False

        info.status = "failed"  # type: ignore
        assert info.is_terminal is True

    def test_queue_full_raises(self, sample_image_bytes):
        from printforge.queue import TaskQueue, TaskInput

        queue = TaskQueue(max_workers=1, max_queue_size=1, result_dir=Path(tempfile.mkdtemp()))
        # Submit one task (will sit pending since no workers started)
        task_input = TaskInput(image_bytes=sample_image_bytes, filename="test.jpg")
        queue.submit(task_input)

        # Second submit should fail with queue full
        with pytest.raises(RuntimeError, match="queue is full"):
            queue.submit(task_input)


class TestAsyncEndpoints:
    @pytest.fixture
    def client(self):
        """Create a test client with a fresh queue."""
        from fastapi.testclient import TestClient
        import printforge.queue as queue_mod

        # Patch global queue before importing app
        queue_mod._global_queue = queue_mod.TaskQueue(max_workers=1)

        from printforge.server import app
        yield TestClient(app)

        # Reset
        queue_mod._global_queue = None

    def test_async_submit_returns_202(self, client, sample_image_bytes):
        response = client.post(
            "/api/v2/generate/async",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"format": "stl", "size_mm": 40.0},
        )
        assert response.status_code == 202
        data = response.json()
        assert "task_id" in data
        assert data["task_id"].startswith("task_")
        assert data["status"] == "pending"

    def test_async_submit_validates_image(self, client):
        response = client.post(
            "/api/v2/generate/async",
            files={"image": ("notimage.txt", b"not an image", "text/plain")},
            data={},
        )
        assert response.status_code == 400
        assert "valid image" in response.json()["detail"].lower()

    def test_async_status_not_found(self, client):
        response = client.get("/api/v2/generate/task_nonexistent/status")
        assert response.status_code == 404

    def test_async_status_returns_pending(self, client, sample_image_bytes):
        # Submit first
        post_resp = client.post(
            "/api/v2/generate/async",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={},
        )
        task_id = post_resp.json()["task_id"]

        status_resp = client.get(f"/api/v2/generate/{task_id}/status")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["task_id"] == task_id
        assert data["status"] in ("pending", "running")
        assert "created_at" in data

    def test_async_result_before_done(self, client, sample_image_bytes):
        post_resp = client.post(
            "/api/v2/generate/async",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"backend": "placeholder"},  # fast
        )
        task_id = post_resp.json()["task_id"]

        # May still be pending or running
        result_resp = client.get(f"/api/v2/generate/{task_id}/result")
        # Either 400 (not done) or 404 (already done and cleaned up) is acceptable
        assert result_resp.status_code in (400, 404)

    def test_queue_stats(self, client, sample_image_bytes):
        # Submit a task
        client.post(
            "/api/v2/generate/async",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={},
        )

        resp = client.get("/api/v2/generate/queue/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_tasks" in data
        assert "by_status" in data
        assert data["total_tasks"] >= 1

    def test_async_with_auth_header(self, client, sample_image_bytes):
        response = client.post(
            "/api/v2/generate/async",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={},
            headers={"X-API-Key": "invalid_key"},
        )
        # Invalid key should fail auth
        assert response.status_code == 401
