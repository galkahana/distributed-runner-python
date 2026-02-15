from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from distributed_runner import TaskFailedError, async_process_continuous
from tests.helpers import double, fail_always, identity


class TestAsyncProcessContinuous:
    async def test_basic_accumulation(self) -> None:
        def acc_fn(acc: int, result: int, submit: Callable[[int], None]) -> tuple[int, bool]:
            return (acc + result, True)

        result = await async_process_continuous(
            [1, 2, 3],
            double,
            acc_fn,
            initial_value=0,
            num_workers=2,
        )
        assert result == 12

    async def test_submit_new_tasks(self) -> None:
        submitted_extra: dict[str, bool] = {"done": False}

        def acc_fn(
            acc: list[int], result: int, submit: Callable[[int], None]
        ) -> tuple[list[int], bool]:
            acc = [*acc, result]
            if not submitted_extra["done"]:
                submit(10)
                submitted_extra["done"] = True
            return (acc, True)

        result = await async_process_continuous(
            [1, 2],
            double,
            acc_fn,
            initial_value=[],
            num_workers=2,
        )
        assert sorted(result) == [2, 4, 20]

    async def test_should_continue_false(self) -> None:
        def acc_fn(acc: int, result: int, submit: Callable[[int], None]) -> tuple[int, bool]:
            submit(99)
            return (acc + result, False)

        result = await async_process_continuous(
            [1, 2, 3],
            double,
            acc_fn,
            initial_value=0,
            num_workers=2,
        )
        assert result == 12

    async def test_natural_completion(self) -> None:
        def acc_fn(acc: int, result: int, submit: Callable[[Any], None]) -> tuple[int, bool]:
            return (acc + result, True)

        result = await async_process_continuous(
            [5],
            identity,
            acc_fn,
            initial_value=0,
            num_workers=1,
        )
        assert result == 5

    async def test_fail_fast(self) -> None:
        def acc_fn(acc: int, result: int, submit: Callable[[int], None]) -> tuple[int, bool]:
            return (acc + result, True)

        with pytest.raises(TaskFailedError):
            await async_process_continuous(
                [1, 2, 3],
                fail_always,
                acc_fn,
                initial_value=0,
                num_workers=2,
            )
