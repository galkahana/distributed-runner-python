from __future__ import annotations

from collections.abc import Callable
from typing import Any, overload

from distributed_runner._batch import process
from distributed_runner._stats import Stats
from distributed_runner._types import A, D, O, PartitionTask


@overload
def map_reduce(
    shared_data: D,
    num_partitions: int,
    process_fn: Callable[[PartitionTask[D]], O],
    *,
    num_workers: int | None = ...,
    max_retries: int = ...,
    accumulator_fn: Callable[[A, O], A],
    initial_value: A,
    on_progress: Callable[[Stats], None] | None = ...,
) -> A: ...


@overload
def map_reduce(
    shared_data: D,
    num_partitions: int,
    process_fn: Callable[[PartitionTask[D]], O],
    *,
    num_workers: int | None = ...,
    max_retries: int = ...,
    on_progress: Callable[[Stats], None] | None = ...,
) -> None: ...


def map_reduce(
    shared_data: Any,
    num_partitions: int,
    process_fn: Callable[..., Any],
    *,
    num_workers: int | None = None,
    max_retries: int = 0,
    accumulator_fn: Callable[..., Any] | None = None,
    initial_value: Any = None,
    on_progress: Callable[[Stats], None] | None = None,
) -> Any:
    """Partition-based parallel processing (simplified MapReduce).

    Splits work into num_partitions PartitionTask objects, each containing the
    partition index, total count, and shared_data. Delegates to process().

    Args:
        shared_data: Data shared across all partitions. Must be picklable.
        num_partitions: Number of partitions to split work into.
        process_fn: Function applied to each PartitionTask. Must be a top-level or
            static function (picklable).
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
    tasks = [
        PartitionTask(
            partition_index=i,
            total_partitions=num_partitions,
            shared_data=shared_data,
        )
        for i in range(num_partitions)
    ]

    if accumulator_fn is not None:
        return process(
            tasks,
            process_fn,
            num_workers=num_workers,
            max_retries=max_retries,
            accumulator_fn=accumulator_fn,
            initial_value=initial_value,
            on_progress=on_progress,
        )
    return process(
        tasks,
        process_fn,
        num_workers=num_workers,
        max_retries=max_retries,
        on_progress=on_progress,
    )
