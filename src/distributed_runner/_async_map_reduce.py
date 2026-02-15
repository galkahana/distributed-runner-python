from __future__ import annotations

from collections.abc import Callable
from typing import Any, overload

from distributed_runner._async_batch import async_process
from distributed_runner._types import PartitionTask


@overload
async def async_map_reduce(
    shared_data: Any,
    num_partitions: int,
    process_fn: Callable[..., Any],
    *,
    num_workers: int | None = ...,
    max_retries: int = ...,
    accumulator_fn: Callable[..., Any],
    initial_value: Any,
) -> Any: ...


@overload
async def async_map_reduce(
    shared_data: Any,
    num_partitions: int,
    process_fn: Callable[..., Any],
    *,
    num_workers: int | None = ...,
    max_retries: int = ...,
) -> list[Any]: ...


async def async_map_reduce(
    shared_data: Any,
    num_partitions: int,
    process_fn: Callable[..., Any],
    *,
    num_workers: int | None = None,
    max_retries: int = 0,
    accumulator_fn: Callable[..., Any] | None = None,
    initial_value: Any = None,
) -> Any:
    """Async version of map_reduce(). Delegates to async_process()."""
    tasks = [
        PartitionTask(
            partition_index=i,
            total_partitions=num_partitions,
            shared_data=shared_data,
        )
        for i in range(num_partitions)
    ]

    if accumulator_fn is not None:
        return await async_process(
            tasks,
            process_fn,
            num_workers=num_workers,
            max_retries=max_retries,
            accumulator_fn=accumulator_fn,
            initial_value=initial_value,
        )
    return await async_process(
        tasks,
        process_fn,
        num_workers=num_workers,
        max_retries=max_retries,
    )
