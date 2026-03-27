"""Feishu webhook notification for PrintForge v2.2."""

import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Feishu webhook URL — stored in env or workspace config
FEISHU_WEBHOOK_URL = os.environ.get(
    "PRINTFORGE_FEISHU_WEBHOOK",
    None
)
FEISHU_WEBHOOK_URL_FILE = Path.home() / ".openclaw" / "workspace" / ".feishu_webhook"

# ── Feishu Card Message ────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    success: bool
    model_file: str  # path or URL
    preview_url: Optional[str]
    operation: str
    duration_ms: int
    error: Optional[str] = None
    user_email: Optional[str] = None


def _get_webhook_url() -> Optional[str]:
    if FEISHU_WEBHOOK_URL:
        return FEISHU_WEBHOOK_URL
    if FEISHU_WEBHOOK_URL_FILE.exists():
        return FEISHU_WEBHOOK_URL_FILE.read_text().strip()
    return None


def _build_card(result: GenerationResult) -> dict:
    """Build a Feishu interactive card payload."""
    if result.success:
        header = {"title": {"tag": "plain_text", "content": "✅ 3D 生成完成"}, "template": "green"}
        elements = [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**任务**: {result.operation}\n"
                               f"**耗时**: {result.duration_ms / 1000:.1f}s\n"
                               f"**文件**: `{result.model_file}`"
                }
            }
        ]
        if result.preview_url:
            elements.append({
                "tag": "action",
                "actions": [
                    {
                        "tag": "open_url",
                        "text": {"tag": "plain_text", "content": "🔗 打开预览"},
                        "url": result.preview_url
                    }
                ]
            })
    else:
        header = {"title": {"tag": "plain_text", "content": "❌ 3D 生成失败"}, "template": "red"}
        elements = [
            {
                "tag": "div",
                "text": {
                    "tag": "lark_md",
                    "content": f"**任务**: {result.operation}\n"
                               f"**错误**: {result.error or 'Unknown error'}\n"
                               f"**耗时**: {result.duration_ms / 1000:.1f}s"
                }
            }
        ]
    
    if result.user_email:
        elements.append({
            "tag": "div",
            "text": {"tag": "lark_md", "content": f"*用户*: {result.user_email}"}
        })
    
    return {
        "msg_type": "interactive",
        "card": {
            "header": header,
            "elements": elements
        }
    }


def send_notification(result: GenerationResult) -> bool:
    """Send a Feishu notification for a generation result. Returns True if sent."""
    webhook_url = _get_webhook_url()
    if not webhook_url:
        logger.debug("No Feishu webhook URL configured, skipping notification")
        return False
    
    try:
        payload = _build_card(result)
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("Feishu notification sent successfully")
            return True
        else:
            logger.warning(f"Feishu notification failed: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        logger.exception(f"Feishu notification error: {e}")
        return False
