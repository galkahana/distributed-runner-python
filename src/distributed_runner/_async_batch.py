from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, overload

from distributed_runner._runner_state import RunnerState
from distributed_runner._stats import Stats
from distributed_runner._types import A, I, O
from distributed_runner._worker import _worker_fn


@overload
async def async_process(
    items: Sequence[I],
    process_fn: Callable[[I], O],
    *,
    num_workers: int | None = ...,
    max_retries: int = ...,
    accumulator_fn: Callable[[A, O], A],
    initial_value: A,
    on_progress: Callable[[Stats], None] | None = ...,
) -> A: ...


@overload
async def async_process(
    items: Sequence[I],
    process_fn: Callable[[I], O],
    *,
    num_workers: int | None = ...,
    max_retries: int = ...,
    on_progress: Callable[[Stats], None] | None = ...,
) -> None: ...


async def async_process(
    items: Sequence[Any],
    process_fn: Callable[..., Any],
    *,
    num_workers: int | None = None,
    max_retries: int = 0,
    accumulator_fn: Callable[..., Any] | None = None,
    initial_value: Any = None,
    on_progress: Callable[[Stats], None] | None = None,
) -> Any:
    """Async version of process(). Runs tasks in a ProcessPoolExecutor via asyncio.

    Each item is passed to process_fn in a separate process. On first failure,
    remaining tasks are cancelled and a TaskFailedError is raised.

    Args:
        items: The items to process. Must be picklable for cross-process transfer.
        process_fn: Function applied to each item. Must be a top-level or static
            function (picklable). Receives one item, returns a result.
        num_workers: Max parallel processes. Defaults to os.cpu_count().
        max_retries: Times to retry a failed task with simple retry. Defaults to 0.
        accumulator_fn: If provided, results are folded via accumulator_fn(acc, result).
            Without it, results are discarded and None is returned.
        initial_value: Starting value for the accumulator. Required when accumulator_fn
            is provided.
        on_progress: Optional callback invoked after each task completes with current Stats.

    Returns:
        The final accumulated value, or None if no accumulator_fn was given.

    Raises:
        TaskFailedError: On first task failure (after retries exhausted), with
            stats on submitted/completed/failed counts.
    """
    workers = num_workers if num_workers is not None else os.cpu_count() or 1
    loop = asyncio.get_running_loop()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = _submit_tasks(loop, executor, items, process_fn, max_retries)
        state = await _collect_results(futures, accumulator_fn, initial_value, on_progress)

    return state.accumulator


def _submit_tasks(
    loop: asyncio.AbstractEventLoop,
    executor: ProcessPoolExecutor,
    items: Sequence[Any],
    process_fn: Callable[..., Any],
    max_retries: int,
) -> set[asyncio.Future[Any]]:
    return {
        loop.run_in_executor(executor, _worker_fn, process_fn, item, max_retries) for item in items
    }


async def _collect_results(
    futures: set[asyncio.Future[Any]],
    accumulator_fn: Callable[..., Any] | None,
    initial_value: Any,
    on_progress: Callable[[Stats], None] | None,
) -> RunnerState:
    state = RunnerState(submitted=len(futures), accumulator=initial_value)
    pending = set(futures)

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        for future in done:
            result = state.get_result_or_fail(future, pending)
            state.completed += 1
            if accumulator_fn is not None:
                state.accumulator = accumulator_fn(state.accumulator, result)
            if on_progress is not None:
                on_progress(state.to_stats())

    return state
