"""Model sharing — public links, embed codes, social sharing for PrintForge v2.2."""

import hashlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, List

SHARES_DIR = Path.home() / ".printforge" / "shares"
SHARES_DIR.mkdir(parents=True, exist_ok=True)
SHARES_FILE = SHARES_DIR / "shares.json"


@dataclass
class SharedModel:
    share_id: str  # short URL-safe ID
    model_id: str
    user_id: str
    created_at: str
    title: str
    description: str
    is_public: bool
    views: int = 0
    likes: int = 0
    downloads: int = 0


def _load_shares() -> Dict[str, dict]:
    if not SHARES_FILE.exists():
        return {}
    with open(SHARES_FILE) as f:
        return json.load(f)


def _save_shares(shares: Dict[str, dict]):
    with open(SHARES_FILE, "w") as f:
        json.dump(shares, f, indent=2)


def create_share(
    model_id: str,
    user_id: str,
    title: str = "",
    description: str = "",
    is_public: bool = True,
) -> SharedModel:
    """Create a public share link for a model."""
    # Generate short share ID from model_id
    raw = f"{model_id}:{user_id}:{time.time()}"
    share_id = hashlib.sha256(raw.encode()).hexdigest()[:10]

    share = SharedModel(
        share_id=share_id,
        model_id=model_id,
        user_id=user_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        title=title or f"3D Model {model_id[:8]}",
        description=description,
        is_public=is_public,
    )

    shares = _load_shares()
    shares[share_id] = asdict(share)
    _save_shares(shares)

    return share


def get_share(share_id: str) -> Optional[dict]:
    """Get a shared model by share ID."""
    shares = _load_shares()
    share = shares.get(share_id)
    if share:
        # Increment views
        share["views"] = share.get("views", 0) + 1
        shares[share_id] = share
        _save_shares(shares)
    return share


def list_public_shares(limit: int = 20) -> List[dict]:
    """List public shared models (for gallery)."""
    shares = _load_shares()
    public = [s for s in shares.values() if s.get("is_public", True)]
    return sorted(public, key=lambda s: s.get("created_at", ""), reverse=True)[:limit]


def like_share(share_id: str) -> bool:
    shares = _load_shares()
    if share_id not in shares:
        return False
    shares[share_id]["likes"] = shares[share_id].get("likes", 0) + 1
    _save_shares(shares)
    return True


def increment_downloads(share_id: str):
    shares = _load_shares()
    if share_id in shares:
        shares[share_id]["downloads"] = shares[share_id].get("downloads", 0) + 1
        _save_shares(shares)


def delete_share(share_id: str) -> bool:
    shares = _load_shares()
    if share_id not in shares:
        return False
    del shares[share_id]
    _save_shares(shares)
    return True


def get_embed_code(share_id: str, base_url: str = "http://localhost:8000") -> str:
    """Generate HTML embed code for a shared model."""
    return f'<iframe src="{base_url}/embed/{share_id}" width="600" height="400" frameborder="0" allowfullscreen></iframe>'
