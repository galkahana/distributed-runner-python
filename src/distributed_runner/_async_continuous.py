from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from distributed_runner._errors import TaskFailedError
from distributed_runner._stats import Stats
from distributed_runner._worker import _worker_fn


async def async_process_continuous(
    initial_tasks: list[Any],
    process_fn: Callable[..., Any],
    accumulator_fn: Callable[..., Any],
    initial_value: Any,
    *,
    num_workers: int | None = None,
    max_retries: int = 0,
) -> Any:
    """Async version of process_continuous(). Uses asyncio.wait(FIRST_COMPLETED)."""
    workers = num_workers if num_workers is not None else os.cpu_count() or 1
    submitted = 0
    completed = 0
    failed = 0
    accumulator = initial_value
    accepting_new = True

    loop = asyncio.get_running_loop()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending: set[asyncio.Future[Any]] = set()

        # Buffer for submissions made during accumulator_fn calls.
        buffered: list[asyncio.Future[Any]] = []

        def submit_fn(item: Any) -> None:
            nonlocal submitted
            if accepting_new:
                future = loop.run_in_executor(executor, _worker_fn, process_fn, item, max_retries)
                buffered.append(future)
                submitted += 1

        for task in initial_tasks:
            future = loop.run_in_executor(executor, _worker_fn, process_fn, task, max_retries)
            pending.add(future)
            submitted += 1

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
                buffered.clear()
                accumulator, should_continue = accumulator_fn(accumulator, result, submit_fn)
                if should_continue:
                    pending.update(buffered)
                else:
                    for f in buffered:
                        f.cancel()
                        submitted -= 1
                    accepting_new = False
                buffered.clear()

    return accumulator
