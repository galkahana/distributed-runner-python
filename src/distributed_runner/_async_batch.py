from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, overload

from distributed_runner._errors import TaskFailedError
from distributed_runner._stats import Stats
from distributed_runner._worker import _worker_fn


@overload
async def async_process(
    items: Sequence[Any],
    process_fn: Callable[..., Any],
    *,
    num_workers: int | None = ...,
    max_retries: int = ...,
    accumulator_fn: Callable[..., Any],
    initial_value: Any,
) -> Any: ...


@overload
async def async_process(
    items: Sequence[Any],
    process_fn: Callable[..., Any],
    *,
    num_workers: int | None = ...,
    max_retries: int = ...,
) -> list[Any]: ...


async def async_process(
    items: Sequence[Any],
    process_fn: Callable[..., Any],
    *,
    num_workers: int | None = None,
    max_retries: int = 0,
    accumulator_fn: Callable[..., Any] | None = None,
    initial_value: Any = None,
) -> Any:
    """Async version of process(). Runs tasks in a ProcessPoolExecutor via asyncio."""
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

    loop = asyncio.get_running_loop()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures: set[asyncio.Future[Any]] = set()
        for item in items:
            future = loop.run_in_executor(executor, _worker_fn, process_fn, item, max_retries)
            futures.add(future)
            submitted += 1

        pending = set(futures)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            for future in done:
                try:
                    result = future.result()
                except Exception as exc:
                    failed += 1
                    stats = Stats(submitted=submitted, completed=completed, failed=failed)
                    for f in pending:
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

    if use_accumulator:
        return accumulator
    return results
