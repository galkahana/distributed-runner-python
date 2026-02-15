from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import Any

from distributed_runner._errors import TaskFailedError
from distributed_runner._stats import Stats
from distributed_runner._worker import _worker_fn


def process_continuous(
    initial_tasks: list[Any],
    process_fn: Callable[..., Any],
    accumulator_fn: Callable[..., Any],
    initial_value: Any,
    *,
    num_workers: int | None = None,
    max_retries: int = 0,
) -> Any:
    """Accumulator-driven continuous processing.

    The accumulator_fn receives (acc, result, submit_fn) and returns (new_acc, should_continue).
    submit_fn can be called to add new tasks. Processing stops when should_continue is False
    and all pending futures complete, or when there are no more pending futures.
    """
    workers = num_workers if num_workers is not None else os.cpu_count() or 1
    submitted = 0
    completed = 0
    failed = 0
    accumulator = initial_value
    accepting_new = True

    with ProcessPoolExecutor(max_workers=workers) as executor:
        pending: set[Future[Any]] = set()

        # Buffer for submissions made during accumulator_fn calls.
        # We only commit them to pending if should_continue is True.
        buffered: list[Future[Any]] = []

        def submit_fn(item: Any) -> None:
            nonlocal submitted
            if accepting_new:
                future = executor.submit(_worker_fn, process_fn, item, max_retries)
                buffered.append(future)
                submitted += 1

        # Submit initial tasks
        for task in initial_tasks:
            future = executor.submit(_worker_fn, process_fn, task, max_retries)
            pending.add(future)
            submitted += 1

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)

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
                    # Cancel any futures submitted during this accumulator call
                    for f in buffered:
                        f.cancel()
                        submitted -= 1
                    accepting_new = False
                buffered.clear()

    return accumulator
