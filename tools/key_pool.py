"""Environment-based API key loading for concurrent experiments."""

from __future__ import annotations

import os


def load_key_pool(pool_variable: str, fallback_variable: str = "LLM_API_KEY") -> list[str]:
    """Load comma-separated keys without storing credentials in source files."""
    raw_pool = os.environ.get(pool_variable, "")
    keys = [key.strip() for key in raw_pool.split(",") if key.strip()]
    if keys:
        return keys

    fallback = os.environ.get(fallback_variable, "").strip()
    return [fallback] if fallback else []
