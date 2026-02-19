from __future__ import annotations

import pytest

from distributed_runner import TaskFailedError, async_process
from tests.helpers import double, fail_always, square


class TestAsyncProcess:
    async def test_basic_no_accumulator(self) -> None:
        result = await async_process([1, 2, 3, 4], double, num_workers=2)
        assert result is None

    async def test_with_accumulator(self) -> None:
        result = await async_process(
            [1, 2, 3, 4],
            double,
            num_workers=2,
            accumulator_fn=lambda acc, x: acc + x,
            initial_value=0,
        )
        assert result == 20

    async def test_empty_list_no_accumulator(self) -> None:
        result = await async_process([], double, num_workers=2)
        assert result is None

    async def test_empty_list_with_accumulator(self) -> None:
        result = await async_process(
            [],
            double,
            num_workers=2,
            accumulator_fn=lambda acc, x: acc + x,
            initial_value=42,
        )
        assert result == 42

    async def test_single_item(self) -> None:
        result = await async_process(
            [7],
            square,
            num_workers=1,
            accumulator_fn=lambda acc, x: acc + [x],
            initial_value=[],
        )
        assert result == [49]

    async def test_fail_fast(self) -> None:
        with pytest.raises(TaskFailedError) as exc_info:
            await async_process([1, 2, 3], fail_always, num_workers=2)
        assert exc_info.value.stats.failed >= 1
