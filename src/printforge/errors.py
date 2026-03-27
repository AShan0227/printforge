"""Standardized error codes and messages for PrintForge API v2.2."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class APIError:
    code: str
    message: str
    status: int
    detail: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"error": {"code": self.code, "message": self.message}}
        if self.detail:
            d["error"]["detail"] = self.detail
        return d


# Auth errors
AUTH_MISSING_KEY = APIError("auth.missing_key", "API key required", 401)
AUTH_INVALID_KEY = APIError("auth.invalid_key", "Invalid or revoked API key", 401)
AUTH_EXPIRED_TOKEN = APIError("auth.expired_token", "JWT token has expired", 401)
AUTH_INVALID_TOKEN = APIError("auth.invalid_token", "Invalid JWT token", 401)

# Quota errors
QUOTA_EXHAUSTED = APIError("quota.exhausted", "API key quota exhausted. Upgrade your plan.", 429)
RATE_LIMITED = APIError("rate.limited", "Too many requests. Try again later.", 429)

# Input errors
INPUT_NOT_IMAGE = APIError("input.not_image", "Uploaded file is not a valid image", 400)
INPUT_TOO_LARGE = APIError("input.too_large", "File exceeds maximum size (50MB)", 400)
INPUT_INVALID_FORMAT = APIError("input.invalid_format", "Unsupported output format", 400)
INPUT_INVALID_BACKEND = APIError("input.invalid_backend", "Unsupported inference backend", 400)
INPUT_MISSING_FIELD = APIError("input.missing_field", "Required field missing", 400)

# Generation errors
GEN_INFERENCE_FAILED = APIError("gen.inference_failed", "3D inference failed", 500)
GEN_ALL_BACKENDS_FAILED = APIError("gen.all_backends_failed", "All inference backends unavailable", 503)
GEN_TIMEOUT = APIError("gen.timeout", "Generation timed out", 504)
GEN_PIPELINE_ERROR = APIError("gen.pipeline_error", "Pipeline processing error", 500)

# Model errors
MODEL_NOT_FOUND = APIError("model.not_found", "Model not found", 404)
MODEL_FILE_MISSING = APIError("model.file_missing", "Model file no longer exists on disk", 404)

# Task errors
TASK_NOT_FOUND = APIError("task.not_found", "Task not found", 404)
TASK_NOT_DONE = APIError("task.not_done", "Task has not completed yet", 409)

# Share errors
SHARE_NOT_FOUND = APIError("share.not_found", "Shared model not found", 404)

# Server errors
SERVER_ERROR = APIError("server.internal", "Internal server error", 500)
SERVER_UNAVAILABLE = APIError("server.unavailable", "Service temporarily unavailable", 503)


# Helper for FastAPI
def error_response(error: APIError, detail: str = "") -> dict:
    """Create a FastAPI-compatible error dict."""
    e = APIError(error.code, error.message, error.status, detail or error.detail)
    return e.to_dict()
