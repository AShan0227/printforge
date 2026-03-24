"""Shared test configuration — clears the global image cache before each test session."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clear_global_cache():
    """Ensure no stale cache entries affect test results."""
    cache_dir = Path.home() / ".printforge" / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    yield
    # Clean up after as well
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
