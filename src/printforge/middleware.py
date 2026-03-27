"""FastAPI middleware for PrintForge v2.2 — logging, error handling, CORS."""

import json
import logging
import time
import traceback
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

from .metrics import get_metrics
from .rate_limit import api_key_limiter, ip_limiter

logger = logging.getLogger("printforge.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with timing, status, and request ID."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.time()

        # Inject request ID
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            logger.error(
                f"{request_id} {request.method} {request.url.path} → 500 ({elapsed:.0f}ms) ERR: {e}"
            )
            return JSONResponse(
                {"error": "Internal server error", "request_id": request_id},
                status_code=500,
            )

        elapsed = (time.time() - start) * 1000
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(int(elapsed))

        # Log
        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        api_key = request.headers.get("X-API-Key", "")
        key_preview = f" key={api_key[:12]}..." if api_key else ""
        logger.log(
            level,
            f"{request_id} {request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms){key_preview}",
        )

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Apply rate limiting to API endpoints."""

    RATE_LIMITED_PATHS = {"/api/generate", "/api/v2/generate/async", "/api/v2/generate/batch"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path

        # Only rate-limit specific endpoints
        if request.method == "POST" and path in self.RATE_LIMITED_PATHS:
            api_key = request.headers.get("X-API-Key", "")
            
            if api_key:
                allowed, retry_after = api_key_limiter.check(api_key)
            else:
                client_ip = request.client.host if request.client else "unknown"
                allowed, retry_after = ip_limiter.check(client_ip)

            if not allowed:
                return JSONResponse(
                    {"error": "Rate limit exceeded", "retry_after": retry_after},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )

        return await call_next(request)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions and return clean JSON errors."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            request_id = getattr(request.state, "request_id", "?")
            logger.exception(f"Unhandled error in {request.method} {request.url.path}")
            return JSONResponse(
                {
                    "error": str(e),
                    "type": type(e).__name__,
                    "request_id": request_id,
                },
                status_code=500,
            )
