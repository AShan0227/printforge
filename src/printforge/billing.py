"""Usage tracking and billing for PrintForge v2.2."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .api_v2 import validate_api_key, increment_usage, get_key_stats

BILLING_DIR = Path.home() / ".openclaw" / "workspace" / "data" / "printforge"
BILLING_DIR.mkdir(parents=True, exist_ok=True)
USAGE_FILE = BILLING_DIR / "usage.json"

PRICING = {
    "free": {"generations": 100, "price_usd": 0},
    "starter": {"generations": 500, "price_usd": 9},
    "pro": {"generations": 2000, "price_usd": 29},
    "enterprise": {"generations": -1, "price_usd": 99},  # unlimited
}


@dataclass
class UsageRecord:
    key: str  # API key
    user_id: str
    timestamp: str
    operation: str  # generate_3d | analyze | optimize | batch
    success: bool
    duration_ms: int
    model_used: str


def record_usage(raw_key: str, operation: str, success: bool, duration_ms: int, model_used: str = "unknown"):
    """Record a generation event and increment usage counter."""
    key = validate_api_key(raw_key)
    if not key:
        return

    record = UsageRecord(
        key=raw_key,
        user_id=key.user_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        operation=operation,
        success=success,
        duration_ms=duration_ms,
        model_used=model_used,
    )

    usage = _load_usage()
    if key.user_id not in usage:
        usage[key.user_id] = []
    usage[key.user_id].append(asdict(record))
    _save_usage(usage)

    # Also increment the key's generation counter
    increment_usage(raw_key)


def get_usage_history(raw_key: str, limit: int = 50) -> list:
    """Get recent usage history for a key."""
    key = validate_api_key(raw_key)
    if not key:
        return []
    
    usage = _load_usage()
    records = usage.get(key.user_id, [])
    return records[-limit:]


def get_monthly_usage(raw_key: str) -> Dict:
    """Get this month's usage statistics."""
    key = validate_api_key(raw_key)
    if not key:
        return {}
    
    usage = _load_usage()
    records = usage.get(key.user_id, [])
    
    now = datetime.now(timezone.utc)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    monthly = [
        r for r in records
        if datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")) >= month_start
    ]
    
    success_count = sum(1 for r in monthly if r["success"])
    fail_count = len(monthly) - success_count
    total_duration = sum(r["duration_ms"] for r in monthly)
    
    return {
        "month": now.strftime("%Y-%m"),
        "total_generations": len(monthly),
        "success": success_count,
        "failed": fail_count,
        "avg_duration_ms": int(total_duration / len(monthly)) if monthly else 0,
        "total_duration_ms": total_duration,
    }


def _load_usage() -> Dict:
    if not USAGE_FILE.exists():
        return {}
    with open(USAGE_FILE) as f:
        return json.load(f)

def _save_usage(usage: Dict):
    with open(USAGE_FILE, "w") as f:
        json.dump(usage, f, indent=2)
