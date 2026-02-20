from __future__ import annotations

from collections.abc import Callable
from typing import Any


def execute_with_retry(fn: Callable[..., Any], item: Any, max_retries: int) -> Any:
    """Execute fn(item) with simple retry. Runs inside the worker process."""
    last_exc: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            return fn(item)
        except Exception as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def worker_fn(fn: Callable[..., Any], item: Any, max_retries: int) -> Any:
    """Top-level picklable function that runs in worker processes."""
    return execute_with_retry(fn, item, max_retries)
