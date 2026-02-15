from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from distributed_runner import TaskFailedError, process_continuous
from tests.helpers import double, fail_always, identity


class TestProcessContinuous:
    def test_basic_accumulation(self) -> None:
        """Process initial tasks and accumulate results."""

        def acc_fn(acc: int, result: int, submit: Callable[[int], None]) -> tuple[int, bool]:
            return (acc + result, True)

        result = process_continuous(
            [1, 2, 3],
            double,
            acc_fn,
            initial_value=0,
            num_workers=2,
        )
        assert result == 12  # 2+4+6

    def test_submit_new_tasks(self) -> None:
        """Accumulator can submit new tasks."""
        submitted_extra: dict[str, bool] = {"done": False}

        def acc_fn(
            acc: list[int], result: int, submit: Callable[[int], None]
        ) -> tuple[list[int], bool]:
            acc = [*acc, result]
            if not submitted_extra["done"]:
                submit(10)
                submitted_extra["done"] = True
            return (acc, True)

        result = process_continuous(
            [1, 2],
            double,
            acc_fn,
            initial_value=[],
            num_workers=2,
        )
        assert sorted(result) == [2, 4, 20]

    def test_should_continue_false_stops(self) -> None:
        """When should_continue is False, no new tasks are accepted."""
        call_count = 0

        def acc_fn(acc: int, result: int, submit: Callable[[int], None]) -> tuple[int, bool]:
            nonlocal call_count
            call_count += 1
            # Try to submit after saying stop - should be ignored
            submit(99)
            return (acc + result, False)

        result = process_continuous(
            [1, 2, 3],
            double,
            acc_fn,
            initial_value=0,
            num_workers=2,
        )
        # All initial tasks complete, but submitted 99 should be ignored after first False
        assert result == 12  # 2+4+6

    def test_natural_completion(self) -> None:
        """Processing stops naturally when no pending futures remain."""

        def acc_fn(acc: int, result: int, submit: Callable[[Any], None]) -> tuple[int, bool]:
            return (acc + result, True)

        result = process_continuous(
            [5],
            identity,
            acc_fn,
            initial_value=0,
            num_workers=1,
        )
        assert result == 5

    def test_fail_fast(self) -> None:
        def acc_fn(acc: int, result: int, submit: Callable[[int], None]) -> tuple[int, bool]:
            return (acc + result, True)

        with pytest.raises(TaskFailedError) as exc_info:
            process_continuous(
                [1, 2, 3],
                fail_always,
                acc_fn,
                initial_value=0,
                num_workers=2,
            )
        assert exc_info.value.stats.failed >= 1
