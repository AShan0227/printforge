"""
PrintForge API v2 — API Key Authentication System
================================================
- API Key generation and validation
- User registration / login with simple JWT
- Per-Key usage quotas (generation_count, quota_limit)
- JSON file storage (users.json)
"""

import hashlib
import json
import os
import secrets
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import jwt

# ── Paths ──────────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_USERS_FILE = _DATA_DIR / "users.json"

# ── JWT Config ─────────────────────────────────────────────────────────────────

JWT_SECRET = os.environ.get("PRINTFORGE_JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24 * 30  # 30 days

# ── Quota Defaults ─────────────────────────────────────────────────────────────

DEFAULT_QUOTA_LIMIT = 100  # generations per API key by default


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class APIKey:
    key: str
    user_id: str
    name: str
    created_at: float = field(default_factory=time.time)
    generation_count: int = 0
    quota_limit: int = DEFAULT_QUOTA_LIMIT
    is_active: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "APIKey":
        return cls(**d)

    @property
    def remaining(self) -> int:
        return max(0, self.quota_limit - self.generation_count)

    @property
    def quota_exhausted(self) -> bool:
        return self.generation_count >= self.quota_limit


@dataclass
class User:
    user_id: str
    username: str
    password_hash: str
    email: str = ""
    created_at: float = field(default_factory=time.time)
    api_keys: list[APIKey] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["api_keys"] = [k.to_dict() for k in self.api_keys]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "User":
        d = dict(d)
        d["api_keys"] = [APIKey.from_dict(k) for k in d.get("api_keys", [])]
        return cls(**d)


# ── Storage ────────────────────────────────────────────────────────────────────

def _ensure_data_dir():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_users() -> dict[str, User]:
    _ensure_data_dir()
    if not _USERS_FILE.exists():
        return {}
    try:
        data = json.loads(_USERS_FILE.read_text())
        return {uid: User.from_dict(u) for uid, u in data.items()}
    except (json.JSONDecodeError, TypeError, KeyError):
        return {}


def _save_users(users: dict[str, User]):
    _ensure_data_dir()
    data = {uid: u.to_dict() for uid, u in users.items()}
    _USERS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
    return h.hex(), salt


def _make_key() -> str:
    return f"pf_{secrets.token_urlsafe(32)}"


def _make_user_id() -> str:
    return f"u_{secrets.token_urlsafe(16)}"


# ── User Management ─────────────────────────────────────────────────────────────

def register_user(username: str, password: str, email: str = "") -> tuple[User, str]:
    """Register a new user. Returns (User, api_key). Raises ValueError on conflict."""
    users = _load_users()
    if any(u.username == username for u in users.values()):
        raise ValueError(f"Username '{username}' already taken")
    if email and any(u.email == email for u in users.values() if u.email):
        raise ValueError(f"Email '{email}' already registered")

    pwd_hash, salt = _hash_password(password)
    user_id = _make_user_id()
    api_key = _make_key()

    user = User(
        user_id=user_id,
        username=username,
        password_hash=f"{pwd_hash}${salt}",
        email=email,
    )
    user.api_keys.append(APIKey(key=api_key, user_id=user_id, name="Default Key"))
    users[user_id] = user
    _save_users(users)
    return user, api_key


def authenticate_user(username: str, password: str) -> Optional[str]:
    """Authenticate a user. Returns JWT token on success, None on failure."""
    users = _load_users()
    for user in users.values():
        if user.username == username:
            pwd_hash, salt = _hash_password(password, salt=user.password_hash.split("$")[1])
            expected = user.password_hash.split("$")[0]
            if pwd_hash == expected:
                return _make_jwt(user.user_id, username)
    return None


def login_user(username: str, password: str) -> tuple[User, str, str]:
    """Login a user. Returns (User, api_key, jwt_token). Raises ValueError on failure."""
    token = authenticate_user(username, password)
    if not token:
        raise ValueError("Invalid username or password")
    users = _load_users()
    user = next((u for u in users.values() if u.username == username), None)
    if not user:
        raise ValueError("User not found")
    api_key = user.api_keys[0].key if user.api_keys else _make_key()
    return user, api_key, token


def _make_jwt(user_id: str, username: str) -> str:
    payload = {
        "sub": user_id,
        "username": username,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY_HOURS * 3600,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_jwt(token: str) -> Optional[dict]:
    """Decode and validate a JWT. Returns payload dict or None."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None


# ── API Key Management ──────────────────────────────────────────────────────────

def create_api_key(user_id: str, name: str = "", quota_limit: int = DEFAULT_QUOTA_LIMIT) -> str:
    """Create a new API key for an existing user. Returns the new key."""
    users = _load_users()
    if user_id not in users:
        raise ValueError(f"User '{user_id}' not found")
    key = _make_key()
    users[user_id].api_keys.append(
        APIKey(key=key, user_id=user_id, name=name or f"Key {len(users[user_id].api_keys)+1}", quota_limit=quota_limit)
    )
    _save_users(users)
    return key


def revoke_api_key(user_id: str, key: str) -> bool:
    """Revoke (deactivate) an API key. Returns True if found and revoked."""
    users = _load_users()
    for user in users.values():
        for api_key in user.api_keys:
            if api_key.key == key:
                api_key.is_active = False
                _save_users(users)
                return True
    return False


def validate_api_key(key: str) -> Optional[APIKey]:
    """Validate an API key. Returns APIKey if valid and active, None otherwise."""
    if not key or not key.startswith("pf_"):
        return None
    users = _load_users()
    for user in users.values():
        for api_key in user.api_keys:
            if api_key.key == key and api_key.is_active:
                return api_key
    return None


def get_user_by_api_key(key: str) -> Optional[User]:
    """Get the User owning an API key."""
    api_key = validate_api_key(key)
    if not api_key:
        return None
    users = _load_users()
    return users.get(api_key.user_id)


def increment_usage(key: str) -> bool:
    """Increment generation count for an API key. Returns True if incremented, False if quota exhausted."""
    api_key = validate_api_key(key)
    if not api_key:
        return False
    if api_key.quota_exhausted:
        return False
    users = _load_users()
    for user in users.values():
        for ak in user.api_keys:
            if ak.key == key:
                ak.generation_count += 1
                _save_users(users)
                return True
    return False


def set_quota(key: str, quota_limit: int) -> bool:
    """Set quota limit for an API key."""
    users = _load_users()
    for user in users.values():
        for ak in user.api_keys:
            if ak.key == key:
                ak.quota_limit = quota_limit
                _save_users(users)
                return True
    return False


def list_user_keys(user_id: str) -> list[APIKey]:
    """List all API keys for a user."""
    users = _load_users()
    user = users.get(user_id)
    if not user:
        return []
    return user.api_keys


def get_key_stats(key: str) -> Optional[dict]:
    """Get usage stats for an API key."""
    api_key = validate_api_key(key)
    if not api_key:
        return None
    return {
        "key": api_key.key,
        "name": api_key.name,
        "generation_count": api_key.generation_count,
        "quota_limit": api_key.quota_limit,
        "remaining": api_key.remaining,
        "quota_exhausted": api_key.quota_exhausted,
        "is_active": api_key.is_active,
    }
