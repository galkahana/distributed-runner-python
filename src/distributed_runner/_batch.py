from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from typing import Any, overload

from distributed_runner._errors import TaskFailedError
from distributed_runner._stats import Stats
from distributed_runner._worker import _worker_fn


@overload
def process(
    items: Sequence[Any],
    process_fn: Callable[..., Any],
    *,
    num_workers: int | None = ...,
    max_retries: int = ...,
    accumulator_fn: Callable[..., Any],
    initial_value: Any,
) -> Any: ...


@overload
def process(
    items: Sequence[Any],
    process_fn: Callable[..., Any],
    *,
    num_workers: int | None = ...,
    max_retries: int = ...,
) -> list[Any]: ...


def process(
    items: Sequence[Any],
    process_fn: Callable[..., Any],
    *,
    num_workers: int | None = None,
    max_retries: int = 0,
    accumulator_fn: Callable[..., Any] | None = None,
    initial_value: Any = None,
) -> Any:
    """Process a list of items in parallel using multiple processes.

    If accumulator_fn is provided, results are folded using accumulator_fn(acc, result).
    Otherwise, returns a list of results (order not guaranteed).
    """
    if not items:
        if accumulator_fn is not None:
            return initial_value
        return []

    workers = num_workers if num_workers is not None else os.cpu_count() or 1
    submitted = 0
    completed = 0
    failed = 0

    use_accumulator = accumulator_fn is not None
    accumulator = initial_value
    results: list[Any] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_index: dict[Future[Any], int] = {}
        for i, item in enumerate(items):
            future = executor.submit(_worker_fn, process_fn, item, max_retries)
            future_to_index[future] = i
            submitted += 1

        try:
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                except Exception as exc:
                    failed += 1
                    stats = Stats(submitted=submitted, completed=completed, failed=failed)
                    # Cancel remaining futures
                    for f in future_to_index:
                        f.cancel()
                    raise TaskFailedError(
                        f"Task failed: {exc}",
                        original_exception=exc,
                        stats=stats,
                    ) from exc

                completed += 1
                if use_accumulator:
                    assert accumulator_fn is not None
                    accumulator = accumulator_fn(accumulator, result)
                else:
                    results.append(result)
        except TaskFailedError:
            raise

    if use_accumulator:
        return accumulator
    return results
