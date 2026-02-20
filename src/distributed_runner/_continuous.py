from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from typing import Any, cast

from distributed_runner._runner_state import RunnerState
from distributed_runner._stats import Stats
from distributed_runner._types import A, I, O
from distributed_runner._worker import worker_fn


def process_continuous(
    initial_tasks: Sequence[I],
    process_fn: Callable[[I], O],
    accumulator_fn: Callable[[A, O, Callable[[I], None]], tuple[A, bool]],
    initial_value: A,
    *,
    num_workers: int | None = None,
    max_retries: int = 0,
    on_progress: Callable[[Stats], None] | None = None,
) -> A:
    """Accumulator-driven continuous processing.

    The accumulator_fn receives (acc, result, submit_fn) and returns
    (new_acc, should_continue). submit_fn can be called to add new tasks dynamically.
    Processing stops when should_continue is False and all pending futures complete,
    or when there are no more pending futures.

    Args:
        initial_tasks: Seed tasks to start processing. Must be picklable.
        process_fn: Function applied to each task. Must be a top-level or static
            function (picklable). Receives one item, returns a result.
        accumulator_fn: Called with (acc, result, submit_fn). Returns (new_acc,
            should_continue). Call submit_fn(item) to enqueue new tasks.
        initial_value: Starting value for the accumulator.
        num_workers: Max parallel processes. Defaults to os.cpu_count().
        max_retries: Times to retry a failed task with simple retry. Defaults to 0.
        on_progress: Optional callback invoked after each task completes with current Stats.

    Returns:
        The final accumulated value.

    Example:
        For fire-and-forget processing (no accumulation, just run all tasks)::

            process_continuous(
                tasks,
                my_fn,
                accumulator_fn=lambda acc, result, submit: (None, True),
                initial_value=None,
            )

    Raises:
        TaskFailedError: On first task failure (after retries exhausted), with
            stats on submitted/completed/failed counts.
    """
    workers = num_workers if num_workers is not None else os.cpu_count() or 1

    with ProcessPoolExecutor(max_workers=workers) as executor:
        state = RunnerState(accumulator=initial_value)
        pending = _submit_initial(executor, initial_tasks, process_fn, max_retries, state)
        _process_loop(
            executor, pending, process_fn, max_retries, accumulator_fn, state, on_progress
        )

    return cast(A, state.accumulator)


def _submit_initial(
    executor: ProcessPoolExecutor,
    initial_tasks: Sequence[Any],
    process_fn: Callable[..., Any],
    max_retries: int,
    state: RunnerState,
) -> set[Future[Any]]:
    pending: set[Future[Any]] = set()
    for task in initial_tasks:
        future = executor.submit(worker_fn, process_fn, task, max_retries)
        pending.add(future)
        state.submitted += 1
    return pending


def _process_loop(
    executor: ProcessPoolExecutor,
    pending: set[Future[Any]],
    process_fn: Callable[..., Any],
    max_retries: int,
    accumulator_fn: Callable[..., Any],
    state: RunnerState,
    on_progress: Callable[[Stats], None] | None,
) -> None:
    buffered: list[Future[Any]] = []
    accepting_new = True

    def submit_fn(item: Any) -> None:
        if accepting_new:
            future = executor.submit(worker_fn, process_fn, item, max_retries)
            buffered.append(future)
            state.submitted += 1

    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)

        for future in done:
            result = state.get_result_or_fail(future, pending)

            state.completed += 1
            accepting_new = _accumulate_and_commit(
                result, accumulator_fn, submit_fn, buffered, pending, state, accepting_new
            )

            if on_progress is not None:
                on_progress(state.to_stats())


def _accumulate_and_commit(
    result: Any,
    accumulator_fn: Callable[..., Any],
    submit_fn: Callable[..., None],
    buffered: list[Future[Any]],
    pending: set[Future[Any]],
    state: RunnerState,
    accepting_new: bool,
) -> bool:
    buffered.clear()
    state.accumulator, should_continue = accumulator_fn(state.accumulator, result, submit_fn)
    if should_continue:
        pending.update(buffered)
    else:
        for f in buffered:
            f.cancel()
            state.submitted -= 1
        accepting_new = False
    buffered.clear()
    return accepting_new
