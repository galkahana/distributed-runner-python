from __future__ import annotations

import functools
import multiprocessing
from unittest.mock import Mock

import pytest

from distributed_runner import Stats, TaskFailedError, process
from tests.helpers import (
    FailOnceThenSucceed,
    barrier_double,
    double,
    fail_always,
    record_item,
    sum_acc,
)


def must_not_be_called(x: int) -> int:
    raise AssertionError("task should not be called for empty input")


@pytest.fixture
def default_items() -> list[int]:
    return [1, 2, 3, 4]


def test_no_accumulator_processes_all_items_returns_none(default_items: list[int]) -> None:
    # Arrange
    with multiprocessing.Manager() as manager:
        calls = manager.list()
        task = functools.partial(record_item, calls)

        # Act
        result = process(default_items, task, num_workers=2)
        recorded = list(calls)

    # Assert
    assert result is None
    assert sorted(recorded) == sorted(default_items)


def test_with_accumulator_sums_results(default_items: list[int]) -> None:
    # Act
    result = process(
        default_items,
        double,
        num_workers=2,
        accumulator_fn=sum_acc,
        initial_value=0,
    )

    # Assert
    assert result == 20  # 2+4+6+8


def test_empty_list_task_not_called_returns_none() -> None:
    # Act
    result = process([], must_not_be_called, num_workers=2)

    # Assert
    assert result is None


def test_empty_list_with_accumulator_task_not_called_returns_initial_value() -> None:
    # Arrange
    initial = 42

    # Act
    result = process(
        [],
        must_not_be_called,
        num_workers=2,
        accumulator_fn=sum_acc,
        initial_value=initial,
    )

    # Assert
    assert result == initial


def test_task_always_fails_raises_task_failed_error() -> None:
    # Arrange
    items = [1, 2, 3]

    # Act / Assert
    with pytest.raises(TaskFailedError) as exc_info:
        process(items, fail_always, num_workers=2)
    assert exc_info.value.stats.failed >= 1
    assert isinstance(exc_info.value.original_exception, ValueError)


def test_task_fails_once_retried_returns_result(
    fail_once_then_succeed: FailOnceThenSucceed,
) -> None:
    # Arrange
    items = [100]

    # Act
    result = process(
        items,
        fail_once_then_succeed,  # fails first attempt, then returns x * 2
        num_workers=1,
        max_retries=2,
        accumulator_fn=sum_acc,
        initial_value=0,
    )

    # Assert
    assert result == 200  # 100 * 2


def test_task_always_fails_retries_exhausted_raises_error() -> None:
    # Act / Assert
    with pytest.raises(TaskFailedError):
        process([1], fail_always, num_workers=1, max_retries=2)


def test_all_workers_receive_tasks() -> None:
    # Arrange — one task per worker; barrier forces all to be active simultaneously
    num_workers = 4
    items = list(range(1, num_workers + 1))

    with multiprocessing.Manager() as manager:
        barrier = manager.Barrier(num_workers)
        task = functools.partial(barrier_double, barrier)

        # Act
        result = process(
            items,
            task,
            num_workers=num_workers,
            accumulator_fn=sum_acc,
            initial_value=0,
        )

    # Assert — reaching here means all workers ran simultaneously (barrier didn't time out)
    assert result == 20  # (1+2+3+4) * 2


def test_on_progress_called_after_each_task(default_items: list[int]) -> None:
    # Arrange
    on_progress = Mock()

    # Act
    process(
        default_items,
        double,
        num_workers=2,
        accumulator_fn=sum_acc,
        initial_value=0,
        on_progress=on_progress,
    )

    # Assert
    assert on_progress.call_count == len(default_items)
    stats_per_call = [c.args[0] for c in on_progress.call_args_list]
    assert [s.completed for s in stats_per_call] == list(range(1, len(default_items) + 1))
    assert stats_per_call[-1] == Stats(
        submitted=len(default_items), completed=len(default_items), failed=0
    )
