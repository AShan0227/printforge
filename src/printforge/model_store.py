"""Model store — track generated 3D models for user history."""

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


STORE_DIR = Path.home() / ".printforge" / "models"
STORE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_FILE = STORE_DIR / "index.json"


@dataclass
class GeneratedModel:
    model_id: str
    user_id: str
    created_at: str
    input_filename: str
    output_path: str
    output_format: str
    vertices: int
    faces: int
    is_watertight: bool
    duration_ms: int
    backend: str
    multi_view: bool
    file_size_bytes: int
    preview_path: Optional[str] = None


def _load_index() -> List[dict]:
    if not INDEX_FILE.exists():
        return []
    with open(INDEX_FILE) as f:
        return json.load(f)


def _save_index(models: List[dict]):
    with open(INDEX_FILE, "w") as f:
        json.dump(models, f, indent=2)


def store_model(
    user_id: str,
    input_filename: str,
    output_path: str,
    output_format: str,
    vertices: int,
    faces: int,
    is_watertight: bool,
    duration_ms: int,
    backend: str = "auto",
    multi_view: bool = False,
) -> GeneratedModel:
    """Store a generated model in the index."""
    model_id = f"mdl_{uuid.uuid4().hex[:12]}"
    
    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    
    # Copy output to persistent store
    store_path = STORE_DIR / model_id
    store_path.mkdir(parents=True, exist_ok=True)
    
    import shutil
    dest = store_path / f"model.{output_format}"
    if os.path.exists(output_path):
        shutil.copy2(output_path, dest)
    
    model = GeneratedModel(
        model_id=model_id,
        user_id=user_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        input_filename=input_filename,
        output_path=str(dest),
        output_format=output_format,
        vertices=vertices,
        faces=faces,
        is_watertight=is_watertight,
        duration_ms=duration_ms,
        backend=backend,
        multi_view=multi_view,
        file_size_bytes=file_size,
    )
    
    index = _load_index()
    index.append(asdict(model))
    _save_index(index)
    
    return model


def list_models(user_id: Optional[str] = None, limit: int = 50) -> List[dict]:
    """List generated models, optionally filtered by user."""
    index = _load_index()
    if user_id:
        index = [m for m in index if m.get("user_id") == user_id]
    return index[-limit:]


def get_model(model_id: str) -> Optional[dict]:
    """Get a specific model by ID."""
    index = _load_index()
    for m in index:
        if m.get("model_id") == model_id:
            return m
    return None


def delete_model(model_id: str) -> bool:
    """Delete a model from the index and disk."""
    index = _load_index()
    new_index = [m for m in index if m.get("model_id") != model_id]
    if len(new_index) == len(index):
        return False
    
    _save_index(new_index)
    
    # Remove files
    model_dir = STORE_DIR / model_id
    if model_dir.exists():
        import shutil
        shutil.rmtree(model_dir)
    
    return True


def get_user_stats(user_id: str) -> dict:
    """Get generation stats for a user."""
    models = list_models(user_id, limit=10000)
    if not models:
        return {"total": 0}
    
    total_duration = sum(m.get("duration_ms", 0) for m in models)
    total_size = sum(m.get("file_size_bytes", 0) for m in models)
    
    return {
        "total": len(models),
        "total_duration_ms": total_duration,
        "avg_duration_ms": int(total_duration / len(models)),
        "total_size_bytes": total_size,
        "formats": list(set(m.get("output_format", "?") for m in models)),
        "multi_view_count": sum(1 for m in models if m.get("multi_view")),
    }
