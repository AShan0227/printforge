"""PrintForge Python SDK — simple client for the PrintForge API.

Usage:
    from printforge.sdk import PrintForge
    
    pf = PrintForge(api_key="pf_xxx", base_url="http://localhost:8000")
    
    # Sync generation
    result = pf.generate("photo.png", format="stl")
    print(result.vertices, result.faces)
    result.save("output.stl")
    
    # Async generation
    task = pf.generate_async("photo.png")
    task.wait()
    task.save("output.stl")
    
    # List models
    models = pf.list_models()
"""

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000"


@dataclass
class GenerationResult:
    vertices: int
    faces: int
    is_watertight: bool
    duration_ms: int
    content: bytes
    format: str

    def save(self, path: str):
        Path(path).write_bytes(self.content)
        logger.info(f"Saved {len(self.content)} bytes to {path}")


@dataclass
class AsyncTask:
    task_id: str
    _client: "PrintForge"

    def status(self) -> dict:
        return self._client._get(f"/api/v2/generate/{self.task_id}/status")

    def wait(self, timeout: int = 300, poll_interval: int = 3) -> dict:
        """Wait for task completion."""
        start = time.time()
        while time.time() - start < timeout:
            s = self.status()
            if s["status"] in ("done", "failed"):
                return s
            time.sleep(poll_interval)
        raise TimeoutError(f"Task {self.task_id} did not complete in {timeout}s")

    def save(self, path: str):
        """Download and save completed result."""
        resp = self._client._session.get(
            f"{self._client.base_url}/api/v2/generate/{self.task_id}/result",
            headers=self._client._headers(),
        )
        resp.raise_for_status()
        Path(path).write_bytes(resp.content)


class PrintForge:
    """PrintForge API client."""

    def __init__(self, api_key: str = "", base_url: str = DEFAULT_BASE_URL):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def _headers(self) -> dict:
        h = {"Accept": "application/json"}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def _get(self, path: str) -> dict:
        resp = self._session.get(f"{self.base_url}{path}", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def _post_json(self, path: str, data: dict) -> dict:
        resp = self._session.post(f"{self.base_url}{path}", json=data, headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    # ── Auth ──────────────────────────────────────────────────────────

    def register(self, username: str, password: str, email: str = "") -> dict:
        return self._post_json("/api/v2/register", {"username": username, "password": password, "email": email})

    def login(self, username: str, password: str) -> dict:
        return self._post_json("/api/v2/login", {"username": username, "password": password})

    def quota(self) -> dict:
        return self._get("/api/v2/quota")

    # ── Generation ────────────────────────────────────────────────────

    def generate(
        self,
        image_path: str,
        format: str = "stl",
        size_mm: float = 50.0,
        backend: str = "auto",
        multi_view: bool = False,
    ) -> GenerationResult:
        """Synchronous 3D generation from image."""
        with open(image_path, "rb") as f:
            resp = self._session.post(
                f"{self.base_url}/api/generate",
                files={"image": (Path(image_path).name, f, "image/png")},
                data={"format": format, "size_mm": size_mm, "backend": backend, "multi_view": multi_view},
                headers={"X-API-Key": self.api_key} if self.api_key else {},
            )
        resp.raise_for_status()
        return GenerationResult(
            vertices=int(resp.headers.get("X-PrintForge-Vertices", 0)),
            faces=int(resp.headers.get("X-PrintForge-Faces", 0)),
            is_watertight=resp.headers.get("X-PrintForge-Watertight", "False") == "True",
            duration_ms=int(resp.headers.get("X-PrintForge-Duration-Ms", 0)),
            content=resp.content,
            format=format,
        )

    def generate_async(
        self,
        image_path: str,
        format: str = "stl",
        backend: str = "auto",
        multi_view: bool = False,
    ) -> AsyncTask:
        """Submit async generation job."""
        with open(image_path, "rb") as f:
            resp = self._session.post(
                f"{self.base_url}/api/v2/generate/async",
                files={"image": (Path(image_path).name, f, "image/png")},
                data={"format": format, "backend": backend, "multi_view": multi_view},
                headers={"X-API-Key": self.api_key} if self.api_key else {},
            )
        resp.raise_for_status()
        data = resp.json()
        return AsyncTask(task_id=data["task_id"], _client=self)

    # ── Models ────────────────────────────────────────────────────────

    def list_models(self, limit: int = 50) -> List[dict]:
        return self._get(f"/api/v2/models?limit={limit}").get("models", [])

    def get_model(self, model_id: str) -> dict:
        return self._get(f"/api/v2/models/{model_id}")

    def download_model(self, model_id: str, save_path: str):
        resp = self._session.get(
            f"{self.base_url}/api/v2/models/{model_id}/download",
            headers=self._headers(),
        )
        resp.raise_for_status()
        Path(save_path).write_bytes(resp.content)

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return self._get("/api/v2/stats")

    def usage(self, limit: int = 50) -> list:
        return self._get(f"/api/v2/usage?limit={limit}").get("usage", [])

    def health(self) -> dict:
        return self._get("/health/detail")

    # ── Webhooks ──────────────────────────────────────────────────────

    def add_webhook(self, url: str, secret: str = "", events: list = None) -> dict:
        return self._post_json("/api/v2/webhooks", {"url": url, "secret": secret, "events": events})

    def list_webhooks(self) -> list:
        return self._get("/api/v2/webhooks").get("webhooks", [])
