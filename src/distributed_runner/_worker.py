from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential


def execute_with_retry(fn: Callable[..., Any], item: Any, max_retries: int) -> Any:
    """Execute fn(item) with tenacity-based retry. Runs inside the worker process."""

    @retry(
        stop=stop_after_attempt(max_retries + 1),
        wait=wait_exponential(multiplier=0.1, max=2),
        reraise=True,
    )
    def _call() -> Any:
        return fn(item)

    return _call()


def _worker_fn(fn: Callable[..., Any], item: Any, max_retries: int) -> Any:
    """Top-level picklable function that runs in worker processes."""
    return execute_with_retry(fn, item, max_retries)
