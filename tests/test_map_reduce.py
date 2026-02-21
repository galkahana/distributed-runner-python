from __future__ import annotations

import functools
import multiprocessing

import pytest

from distributed_runner import TaskFailedError, map_reduce
from tests.helpers import (
    PartitionFailOnceThenSucceed,
    partition_fail,
    partition_sum,
    record_partition_task,
    sum_acc,
)


@pytest.fixture
def hundred_numbers() -> list[int]:
    return list(range(1, 101))


def test_with_accumulator_sums_all_partitions(hundred_numbers: list[int]) -> None:
    # Arrange
    expected = sum(range(1, 101))

    # Act
    result = map_reduce(
        hundred_numbers,
        num_partitions=4,
        process_fn=partition_sum,
        num_workers=2,
        accumulator_fn=sum_acc,
        initial_value=0,
    )

    # Assert
    assert result == expected


def test_no_accumulator_processes_all_partitions_returns_none() -> None:
    # Arrange
    num_partitions = 2
    data = list(range(1, 11))

    with multiprocessing.Manager() as manager:
        recorded = manager.list()
        task = functools.partial(record_partition_task, recorded)

        # Act
        result = map_reduce(data, num_partitions=num_partitions, process_fn=task, num_workers=2)
        captured = list(recorded)

    # Assert
    assert result is None
    assert len(captured) == num_partitions


def test_partitions_created_with_correct_indices() -> None:
    # Arrange
    num_partitions = 4

    with multiprocessing.Manager() as manager:
        recorded = manager.list()
        task = functools.partial(record_partition_task, recorded)

        # Act
        map_reduce(
            list(range(10)),
            num_partitions=num_partitions,
            process_fn=task,
            num_workers=num_partitions,
        )
        captured = list(recorded)

    # Assert
    assert len(captured) == num_partitions
    assert sorted(idx for idx, _ in captured) == list(range(num_partitions))
    assert all(total == num_partitions for _, total in captured)


def test_single_partition_accumulates_correctly() -> None:
    # Arrange
    data = [10, 20, 30]

    # Act
    result = map_reduce(
        data,
        num_partitions=1,
        process_fn=partition_sum,
        num_workers=1,
        accumulator_fn=sum_acc,
        initial_value=0,
    )

    # Assert
    assert result == 60


def test_partition_fails_once_retried_returns_result(
    hundred_numbers: list[int], partition_fail_once_then_succeed: PartitionFailOnceThenSucceed
) -> None:
    # Arrange
    expected = sum(range(1, 101))

    # Act
    result = map_reduce(
        hundred_numbers,
        num_partitions=4,
        # fails first attempt per partition, then returns partition sum
        process_fn=partition_fail_once_then_succeed,
        num_workers=2,
        max_retries=2,
        accumulator_fn=sum_acc,
        initial_value=0,
    )

    # Assert
    assert result == expected


def test_partition_raises_exception_raises_task_failed_error() -> None:
    # Arrange
    data = [1, 2, 3]

    # Act / Assert
    with pytest.raises(TaskFailedError) as exc_info:
        map_reduce(
            data,
            num_partitions=2,
            process_fn=partition_fail,
            num_workers=2,
        )
    assert isinstance(exc_info.value.original_exception, ValueError)
