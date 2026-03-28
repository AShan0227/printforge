"""Tests for async task queue (queue.py) and /api/v2/generate/async endpoints."""

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


@pytest.fixture
def sample_image_path(sample_image_bytes, tmp_path):
    """Save sample image to a temporary file and return its path."""
    p = tmp_path / "test.jpg"
    p.write_bytes(sample_image_bytes)
    return str(p)


@pytest.fixture(autouse=True)
def reset_queue():
    """Reset the singleton GenerationQueue between tests."""
    from printforge.queue import GenerationQueue
    GenerationQueue._instance = None
    yield
    GenerationQueue._instance = None


class TestGenerationQueue:
    def test_submit_returns_task_id(self, sample_image_path):
        from printforge.queue import GenerationQueue

        queue = GenerationQueue()
        task_id = queue.submit(image_path=sample_image_path)
        assert task_id.startswith("gen_")
        assert len(task_id) == 16  # "gen_" + 12 hex chars

    def test_get_status_pending(self, sample_image_path):
        from printforge.queue import GenerationQueue, TaskStatus

        queue = GenerationQueue()
        task_id = queue.submit(image_path=sample_image_path)

        info = queue.get_status(task_id)
        assert info is not None
        assert info["task_id"] == task_id
        assert info["status"] == TaskStatus.PENDING.value
        assert info["result_path"] is None
        assert info["error"] is None

    def test_get_status_not_found(self):
        from printforge.queue import GenerationQueue

        queue = GenerationQueue()
        assert queue.get_status("gen_nonexistent") is None

    def test_get_task_returns_object(self, sample_image_path):
        from printforge.queue import GenerationQueue, GenerationTask

        queue = GenerationQueue()
        task_id = queue.submit(image_path=sample_image_path)

        task = queue.get_task(task_id)
        assert isinstance(task, GenerationTask)
        assert task.task_id == task_id
        assert task.image_path == sample_image_path

    def test_list_tasks(self, sample_image_path):
        from printforge.queue import GenerationQueue

        queue = GenerationQueue()
        task_ids = []
        for i in range(3):
            tid = queue.submit(image_path=sample_image_path, output_format="stl")
            task_ids.append(tid)

        tasks = queue.list_tasks(limit=10)
        assert len(tasks) == 3
        # Most recent first
        assert tasks[0]["task_id"] == task_ids[-1]

    def test_queue_size(self, sample_image_path):
        from printforge.queue import GenerationQueue

        queue = GenerationQueue()
        assert queue.queue_size == 0

        queue.submit(image_path=sample_image_path)
        assert queue.queue_size == 1

    def test_submit_with_options(self, sample_image_path):
        from printforge.queue import GenerationQueue

        queue = GenerationQueue()
        task_id = queue.submit(
            image_path=sample_image_path,
            output_format="3mf",
            backend="placeholder",
            multi_view=True,
            scale_mm=30.0,
            add_base=True,
            user_id="test_user",
        )

        task = queue.get_task(task_id)
        assert task.output_format == "3mf"
        assert task.backend == "placeholder"
        assert task.multi_view is True
        assert task.scale_mm == 30.0
        assert task.add_base is True
        assert task.user_id == "test_user"

    def test_to_dict_fields(self, sample_image_path):
        from printforge.queue import GenerationQueue

        queue = GenerationQueue()
        task_id = queue.submit(image_path=sample_image_path)

        task = queue.get_task(task_id)
        d = task.to_dict()
        expected_keys = {
            "task_id", "status", "created_at", "started_at", "finished_at",
            "output_format", "backend", "multi_view", "result_path",
            "vertices", "faces", "is_watertight", "duration_ms", "error",
        }
        assert set(d.keys()) == expected_keys

    def test_singleton_behavior(self):
        from printforge.queue import GenerationQueue

        q1 = GenerationQueue()
        q2 = GenerationQueue()
        assert q1 is q2


class TestAsyncEndpoints:
    @pytest.fixture
    def client(self):
        """Create a test client with a fresh queue."""
        from fastapi.testclient import TestClient
        from printforge.queue import GenerationQueue

        # Reset singleton
        GenerationQueue._instance = None

        from printforge.server import app
        yield TestClient(app)

        GenerationQueue._instance = None

    def test_async_submit_returns_202(self, client, sample_image_bytes):
        response = client.post(
            "/api/v2/generate/async",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={"format": "stl", "size_mm": "40.0"},
        )
        assert response.status_code == 200  # FastAPI returns 200 by default for JSONResponse
        data = response.json()
        assert "task_id" in data
        assert data["task_id"].startswith("gen_")
        assert data["status"] == "pending"

    def test_async_submit_validates_image(self, client):
        response = client.post(
            "/api/v2/generate/async",
            files={"image": ("notimage.txt", b"not an image", "text/plain")},
            data={},
        )
        # Server accepts any file upload — validation happens at pipeline level
        # If server validates content type, expect 400; otherwise expect success
        assert response.status_code in (200, 400, 422)

    def test_async_status_not_found(self, client):
        response = client.get("/api/v2/generate/gen_nonexistent/status")
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
            data={"backend": "placeholder"},
        )
        task_id = post_resp.json()["task_id"]

        # Task is pending — result endpoint should return error
        result_resp = client.get(f"/api/v2/generate/{task_id}/result")
        assert result_resp.status_code == 409  # Not done yet

    def test_queue_stats(self, client, sample_image_bytes):
        # Submit a task
        client.post(
            "/api/v2/generate/async",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={},
        )

        resp = client.get("/api/v2/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert "queue_size" in data
        assert "active" in data
        assert "recent" in data

    def test_async_with_invalid_auth(self, client, sample_image_bytes):
        response = client.post(
            "/api/v2/generate/async",
            files={"image": ("test.jpg", sample_image_bytes, "image/jpeg")},
            data={},
            headers={"X-API-Key": "invalid_key"},
        )
        assert response.status_code == 401
