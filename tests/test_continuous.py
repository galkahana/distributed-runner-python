from __future__ import annotations

from collections.abc import Callable

import pytest

from distributed_runner import TaskFailedError, process_continuous
from tests.helpers import (
    FailOnceThenSucceed,
    continuous_sum_acc,
    double,
    fail_always,
    identity,
)


def test_initial_tasks_accumulates_results() -> None:
    # Act
    result = process_continuous(
        [1, 2, 3],
        double,
        continuous_sum_acc,
        initial_value=0,
        num_workers=2,
    )

    # Assert
    assert result == 12  # 2+4+6


def test_accumulator_submits_new_tasks_processes_them() -> None:
    # Arrange
    submitted_extra: dict[str, bool] = {"done": False}

    def acc_fn(
        acc: list[int], result: int, submit: Callable[[int], None]
    ) -> tuple[list[int], bool]:
        acc = [*acc, result]
        if not submitted_extra["done"]:
            submit(10)
            submitted_extra["done"] = True
        return (acc, True)

    # Act
    result = process_continuous(
        [1, 2],
        double,
        acc_fn,
        initial_value=[],
        num_workers=2,
    )

    # Assert
    assert sorted(result) == [2, 4, 20]


def test_accumulator_returns_false_stops_accepting_new_tasks() -> None:
    # Arrange
    def acc_fn(acc: int, result: int, submit: Callable[[int], None]) -> tuple[int, bool]:
        submit(99)  # should be ignored after returning False
        return (acc + result, False)

    # Act
    result = process_continuous(
        [1, 2, 3],
        double,
        acc_fn,
        initial_value=0,
        num_workers=2,
    )

    # Assert
    assert result == 12  # 2+4+6, submitted 99 ignored


def test_no_pending_futures_stops_naturally() -> None:
    # Act
    result = process_continuous(
        [5],
        identity,
        continuous_sum_acc,
        initial_value=0,
        num_workers=1,
    )

    # Assert
    assert result == 5


def test_task_fails_once_retried_returns_result(
    fail_once_then_succeed: FailOnceThenSucceed,
) -> None:
    # Act
    result = process_continuous(
        [100],
        fail_once_then_succeed,  # fails first attempt, then returns x * 2
        continuous_sum_acc,
        initial_value=0,
        num_workers=1,
        max_retries=2,
    )

    # Assert
    assert result == 200  # 100 * 2


def test_task_raises_exception_raises_task_failed_error() -> None:
    # Act / Assert
    with pytest.raises(TaskFailedError) as exc_info:
        process_continuous(
            [1, 2, 3],
            fail_always,
            continuous_sum_acc,
            initial_value=0,
            num_workers=2,
        )
    assert exc_info.value.stats.failed >= 1
