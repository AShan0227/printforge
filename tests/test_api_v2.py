"""Tests for API v2 auth, billing, and Feishu notifier."""

import os
import json
import tempfile
import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clean_data_dir(tmp_path, monkeypatch):
    """Use temp dir for all data files."""
    monkeypatch.setattr("printforge.api_v2._DATA_DIR", tmp_path)
    monkeypatch.setattr("printforge.api_v2._USERS_FILE", tmp_path / "users.json")
    monkeypatch.setattr("printforge.billing.BILLING_DIR", tmp_path)
    monkeypatch.setattr("printforge.billing.USAGE_FILE", tmp_path / "usage.json")
    yield


class TestApiKeyManagement:
    def test_register_and_validate(self):
        from printforge.api_v2 import register_user, validate_api_key
        
        user, raw_key = register_user("testuser", "pass123", "test@example.com")
        assert user.username == "testuser"
        assert raw_key.startswith("pf_")
        
        # Validate key
        api_key = validate_api_key(raw_key)
        assert api_key is not None
        assert api_key.user_id == user.user_id

    def test_invalid_key_returns_none(self):
        from printforge.api_v2 import validate_api_key
        
        assert validate_api_key("invalid_key") is None
        assert validate_api_key("pf_invalid") is None
        assert validate_api_key("") is None

    def test_quota_exhaustion(self):
        from printforge.api_v2 import register_user, create_api_key, validate_api_key, increment_usage
        
        user, _ = register_user("quotauser", "pass123")
        key = create_api_key(user.user_id, quota_limit=2)
        
        api_key = validate_api_key(key)
        assert api_key is not None
        assert api_key.remaining == 2
        
        increment_usage(key)
        increment_usage(key)
        
        api_key = validate_api_key(key)
        assert api_key is not None
        assert api_key.quota_exhausted is True

    def test_revoke_key(self):
        from printforge.api_v2 import register_user, revoke_api_key, validate_api_key
        
        user, raw_key = register_user("revokeuser", "pass123")
        
        assert revoke_api_key(user.user_id, raw_key) is True
        assert validate_api_key(raw_key) is None


class TestJWT:
    def test_create_and_decode_jwt(self):
        from printforge.api_v2 import _make_jwt, decode_jwt
        
        token = _make_jwt("usr_123", "testuser")
        payload = decode_jwt(token)
        assert payload is not None
        assert payload["sub"] == "usr_123"
        assert payload["username"] == "testuser"

    def test_invalid_token(self):
        from printforge.api_v2 import decode_jwt
        
        assert decode_jwt("invalid.token.here") is None
        assert decode_jwt("") is None


class TestUserManagement:
    def test_login_user(self):
        from printforge.api_v2 import register_user, login_user
        
        user, _ = register_user("loginuser", "pass123", "test@example.com")
        
        logged_user, token, first_key = login_user("loginuser", "pass123")
        assert logged_user.user_id == user.user_id
        assert token  # JWT token
        assert first_key  # returns first key (could be pf_ or JWT depending on impl)

    def test_login_wrong_password(self):
        from printforge.api_v2 import register_user, authenticate_user
        
        register_user("wrongpw", "correct")
        assert authenticate_user("wrongpw", "wrong") is None

    def test_list_user_keys(self):
        from printforge.api_v2 import register_user, create_api_key, list_user_keys
        
        user, _ = register_user("multikey", "pass123")
        create_api_key(user.user_id, "key2")
        create_api_key(user.user_id, "key3")
        
        keys = list_user_keys(user.user_id)
        assert len(keys) == 3  # 1 from register + 2 new


class TestFeishuNotifier:
    def test_build_card_success(self):
        from printforge.feishu_notifier import _build_card, GenerationResult
        
        result = GenerationResult(
            success=True,
            model_file="/tmp/output.3mf",
            preview_url="http://localhost:8000/preview?url=/tmp/output.glb",
            operation="generate_3d",
            duration_ms=3200,
        )
        
        card = _build_card(result)
        assert card["msg_type"] == "interactive"
        assert "green" in str(card["card"]["header"])

    def test_build_card_failure(self):
        from printforge.feishu_notifier import _build_card, GenerationResult
        
        result = GenerationResult(
            success=False,
            model_file="",
            preview_url=None,
            operation="generate_3d",
            duration_ms=500,
            error="Model inference timeout",
        )
        
        card = _build_card(result)
        assert "red" in str(card["card"]["header"])

    def test_no_webhook_skips(self):
        from printforge.feishu_notifier import send_notification, GenerationResult
        
        result = GenerationResult(
            success=True, model_file="test.3mf", preview_url=None,
            operation="test", duration_ms=100,
        )
        assert send_notification(result) is False
