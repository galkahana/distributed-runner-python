"""Top-level picklable test functions for use with ProcessPoolExecutor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from distributed_runner import PartitionTask


def double(x: int) -> int:
    return x * 2


def fail_always(x: int) -> int:
    raise ValueError(f"always fails: {x}")


class FailOnceThenSucceed:
    """Fails on first call per input value, succeeds on retry returning x * 2."""

    def __init__(self) -> None:
        self._fail_count: dict[str, int] = {}

    def __call__(self, x: int) -> int:
        key = str(x)
        count = self._fail_count.get(key, 0)
        self._fail_count[key] = count + 1
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


class PartitionFailOnceThenSucceed:
    """Fails on first call per partition index, succeeds on retry returning the partition sum."""

    def __init__(self) -> None:
        self._fail_count: dict[str, int] = {}

    def __call__(self, task: PartitionTask[list[int]]) -> int:
        key = str(task.partition_index)
        count = self._fail_count.get(key, 0)
        self._fail_count[key] = count + 1
        if count == 0:
            raise ValueError(f"transient failure: partition {task.partition_index}")
        return partition_sum(task)


def identity(x: int) -> int:
    return x


def record_item(collection: Any, x: int) -> int:
    """Appends x to a multiprocessing-safe collection and returns x."""
    collection.append(x)
    return x


def record_partition_task(collection: Any, task: PartitionTask[Any]) -> int:
    """Records partition_index and total_partitions to a multiprocessing-safe collection."""
    collection.append((task.partition_index, task.total_partitions))
    return 0


def barrier_double(barrier: Any, x: int) -> int:
    """Waits at a multiprocessing barrier then doubles x. Proves all workers run simultaneously."""
    barrier.wait(timeout=5)
    return x * 2


def continuous_sum_acc(acc: int, result: int, submit: Callable[[Any], None]) -> tuple[int, bool]:
    return (acc + result, True)


def sum_acc(acc: int, x: int) -> int:
    return acc + x
