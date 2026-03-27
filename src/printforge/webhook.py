"""Generic webhook notification for PrintForge v2.2."""

import json
import logging
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF = [2, 5, 10]  # seconds


@dataclass
class WebhookEvent:
    event_type: str  # generation.completed | generation.failed | model.deleted
    task_id: str
    user_id: str
    data: Dict[str, Any]
    timestamp: str


@dataclass
class WebhookConfig:
    url: str
    secret: str = ""  # for HMAC signing
    events: List[str] = None  # filter, None = all events

    def __post_init__(self):
        if self.events is None:
            self.events = ["*"]


# In-memory webhook registry (per-user)
_webhooks: Dict[str, List[WebhookConfig]] = {}


def register_webhook(user_id: str, url: str, secret: str = "", events: Optional[List[str]] = None):
    """Register a webhook for a user."""
    if user_id not in _webhooks:
        _webhooks[user_id] = []
    _webhooks[user_id].append(WebhookConfig(url=url, secret=secret, events=events))
    logger.info(f"Webhook registered for {user_id}: {url}")


def unregister_webhook(user_id: str, url: str) -> bool:
    """Unregister a webhook."""
    if user_id not in _webhooks:
        return False
    before = len(_webhooks[user_id])
    _webhooks[user_id] = [w for w in _webhooks[user_id] if w.url != url]
    return len(_webhooks[user_id]) < before


def list_webhooks(user_id: str) -> List[Dict]:
    """List webhooks for a user."""
    return [
        {"url": w.url, "events": w.events}
        for w in _webhooks.get(user_id, [])
    ]


def fire_event(event: WebhookEvent):
    """Fire a webhook event to all registered endpoints (async, non-blocking)."""
    user_hooks = _webhooks.get(event.user_id, [])
    if not user_hooks:
        return

    for hook in user_hooks:
        if hook.events and "*" not in hook.events and event.event_type not in hook.events:
            continue
        # Fire in background thread
        threading.Thread(
            target=_deliver,
            args=(hook, event),
            daemon=True,
        ).start()


def _deliver(hook: WebhookConfig, event: WebhookEvent):
    """Deliver webhook with retries."""
    payload = {
        "event": event.event_type,
        "task_id": event.task_id,
        "user_id": event.user_id,
        "data": event.data,
        "timestamp": event.timestamp,
    }

    headers = {"Content-Type": "application/json"}
    if hook.secret:
        import hashlib
        import hmac
        signature = hmac.new(
            hook.secret.encode(),
            json.dumps(payload, sort_keys=True).encode(),
            hashlib.sha256,
        ).hexdigest()
        headers["X-Webhook-Signature"] = signature

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(hook.url, json=payload, headers=headers, timeout=10)
            if resp.status_code < 300:
                logger.info(f"Webhook delivered: {event.event_type} → {hook.url}")
                return
            logger.warning(f"Webhook {hook.url} returned {resp.status_code}, retry {attempt + 1}/{MAX_RETRIES}")
        except Exception as e:
            logger.warning(f"Webhook {hook.url} failed: {e}, retry {attempt + 1}/{MAX_RETRIES}")

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF[attempt])

    logger.error(f"Webhook delivery failed after {MAX_RETRIES} retries: {hook.url}")
