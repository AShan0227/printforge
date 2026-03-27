"""Sliding window rate limiter for PrintForge API v2."""

import time
import threading
from collections import defaultdict
from typing import Optional, Tuple


class SlidingWindowLimiter:
    """Per-key sliding window rate limiter.
    
    Usage:
        limiter = SlidingWindowLimiter(max_requests=20, window_seconds=60)
        allowed, retry_after = limiter.check("user_key_123")
        if not allowed:
            return 429, {"Retry-After": str(retry_after)}
    """

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, key: str) -> Tuple[bool, int]:
        """Check if request is allowed.
        
        Returns:
            (allowed, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            # Remove expired entries
            self._requests[key] = [
                t for t in self._requests[key] if t > window_start
            ]

            if len(self._requests[key]) >= self.max_requests:
                # Calculate retry-after: time until oldest request expires
                oldest = self._requests[key][0]
                retry_after = int(oldest + self.window_seconds - now) + 1
                return False, max(1, retry_after)

            self._requests[key].append(now)
            return True, 0

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        window_start = now - self.window_seconds
        with self._lock:
            current = [t for t in self._requests.get(key, []) if t > window_start]
            return max(0, self.max_requests - len(current))


# Global limiters
api_key_limiter = SlidingWindowLimiter(max_requests=60, window_seconds=60)   # 60/min per key
ip_limiter = SlidingWindowLimiter(max_requests=10, window_seconds=60)        # 10/min per IP
generate_limiter = SlidingWindowLimiter(max_requests=10, window_seconds=60)  # 10 gen/min per key
