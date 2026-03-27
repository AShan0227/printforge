"""Integration tests for PrintForge v2.2 new features."""
import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestRateLimiter:
    def test_allows_within_limit(self):
        from printforge.rate_limit import SlidingWindowLimiter
        limiter = SlidingWindowLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            ok, _ = limiter.check("test_key")
            assert ok

    def test_blocks_over_limit(self):
        from printforge.rate_limit import SlidingWindowLimiter
        limiter = SlidingWindowLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.check("test_key")
        ok, retry = limiter.check("test_key")
        assert not ok
        assert retry > 0

    def test_remaining(self):
        from printforge.rate_limit import SlidingWindowLimiter
        limiter = SlidingWindowLimiter(max_requests=10, window_seconds=60)
        limiter.check("k1")
        limiter.check("k1")
        assert limiter.get_remaining("k1") == 8


class TestWebhook:
    def test_register_and_list(self):
        from printforge.webhook import register_webhook, list_webhooks, _webhooks
        _webhooks.clear()
        register_webhook("usr1", "https://example.com/hook", events=["generation.done"])
        hooks = list_webhooks("usr1")
        assert len(hooks) == 1
        assert hooks[0]["url"] == "https://example.com/hook"

    def test_unregister(self):
        from printforge.webhook import register_webhook, unregister_webhook, list_webhooks, _webhooks
        _webhooks.clear()
        register_webhook("usr1", "https://example.com/hook")
        assert unregister_webhook("usr1", "https://example.com/hook")
        assert len(list_webhooks("usr1")) == 0


class TestSSE:
    def test_event_bus_publish_subscribe(self):
        import asyncio
        from printforge.sse import EventBus

        bus = EventBus.__new__(EventBus)
        bus._subscribers = {}

        q = bus.subscribe("test")
        bus.publish("test_event", {"foo": "bar"}, "test")

        msg = asyncio.get_event_loop().run_until_complete(asyncio.wait_for(q.get(), timeout=1))
        assert msg["event"] == "test_event"
        assert msg["data"]["foo"] == "bar"


class TestModelStore:
    def test_store_and_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("printforge.model_store.STORE_DIR", tmp_path)
        monkeypatch.setattr("printforge.model_store.INDEX_FILE", tmp_path / "index.json")
        from printforge.model_store import store_model, list_models, get_model

        # Create a dummy file
        dummy = tmp_path / "test.stl"
        dummy.write_bytes(b"solid test")

        m = store_model(
            user_id="usr1", input_filename="photo.png",
            output_path=str(dummy), output_format="stl",
            vertices=100, faces=200, is_watertight=True,
            duration_ms=5000,
        )
        assert m.model_id.startswith("mdl_")

        models = list_models(user_id="usr1")
        assert len(models) == 1

        detail = get_model(m.model_id)
        assert detail is not None
        assert detail["vertices"] == 100


class TestQueue:
    def test_queue_submit(self):
        from printforge.queue import GenerationQueue
        # Reset singleton for test
        GenerationQueue._instance = None
        q = GenerationQueue()
        task_id = q.submit(image_path="/tmp/test.png", output_format="stl")
        assert task_id.startswith("gen_")
        status = q.get_status(task_id)
        assert status["status"] == "pending"


class TestExportGLB:
    def test_export(self, tmp_path):
        import trimesh
        from printforge.export_glb import export_glb
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        out = str(tmp_path / "test.glb")
        export_glb(mesh, out, color="#FF0000")
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


class TestSDK:
    def test_sdk_init(self):
        from printforge.sdk import PrintForge
        pf = PrintForge(api_key="pf_test", base_url="http://localhost:8000")
        assert pf.api_key == "pf_test"
        assert pf.base_url == "http://localhost:8000"


class TestConverter:
    def test_convert_stl_to_obj(self, tmp_path):
        import trimesh
        from printforge.converter import convert_mesh
        
        # Create test STL
        mesh = trimesh.creation.box(extents=[10, 10, 10])
        stl_path = str(tmp_path / "test.stl")
        mesh.export(stl_path)
        
        obj_path = str(tmp_path / "test.obj")
        stats = convert_mesh(stl_path, obj_path)
        assert stats["input_format"] == "stl"
        assert stats["output_format"] == "obj"
        assert os.path.exists(obj_path)

    def test_simplify(self, tmp_path):
        import trimesh
        from printforge.converter import convert_mesh
        
        mesh = trimesh.creation.icosphere(subdivisions=3)
        stl_path = str(tmp_path / "dense.stl")
        mesh.export(stl_path)
        
        out_path = str(tmp_path / "simple.stl")
        stats = convert_mesh(stl_path, out_path, target_faces=100)
        assert stats["output_faces"] <= stats["input_faces"]

    def test_mesh_info(self, tmp_path):
        import trimesh
        from printforge.converter import get_mesh_info
        
        mesh = trimesh.creation.box(extents=[10, 20, 30])
        path = str(tmp_path / "box.stl")
        mesh.export(path)
        
        info = get_mesh_info(path)
        assert info["vertices"] > 0
        assert info["faces"] > 0
        assert info["format"] == "stl"
