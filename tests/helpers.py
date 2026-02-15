"""Top-level picklable test functions for use with ProcessPoolExecutor."""

from __future__ import annotations

import time

from distributed_runner import PartitionTask

_fail_count: dict[str, int] = {}


def double(x: int) -> int:
    return x * 2


def square(x: int) -> int:
    return x * x


def slow_double(x: int) -> int:
    time.sleep(0.1)
    return x * 2


def fail_always(x: int) -> int:
    raise ValueError(f"always fails: {x}")


def fail_once_then_succeed(x: int) -> int:
    """Fails on first call per process, succeeds on retry.

    Uses a module-level dict keyed by input value to track attempts.
    """
    key = str(x)
    count = _fail_count.get(key, 0)
    _fail_count[key] = count + 1
    if count == 0:
        raise ValueError(f"transient failure: {x}")
    return x * 2


def partition_sum(task: PartitionTask[list[int]]) -> int:
    """Sum elements in this partition's slice of the shared data."""
    data = task.shared_data
    chunk_size = len(data) // task.total_partitions
    start = task.partition_index * chunk_size
    end = len(data) if task.partition_index == task.total_partitions - 1 else start + chunk_size
    return sum(data[start:end])


def partition_fail(task: PartitionTask[list[int]]) -> int:
    raise ValueError(f"partition {task.partition_index} failed")


def identity(x: int) -> int:
    return x
