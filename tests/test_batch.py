from __future__ import annotations

import pytest

from distributed_runner import TaskFailedError, process
from tests.helpers import double, fail_always, fail_once_then_succeed, slow_double, square


class TestProcess:
    def test_basic_list_result(self) -> None:
        result = process([1, 2, 3, 4], double, num_workers=2)
        assert sorted(result) == [2, 4, 6, 8]

    def test_with_accumulator(self) -> None:
        result = process(
            [1, 2, 3, 4],
            double,
            num_workers=2,
            accumulator_fn=lambda acc, x: acc + x,
            initial_value=0,
        )
        assert result == 20  # 2+4+6+8

    def test_empty_list_returns_empty(self) -> None:
        result = process([], double, num_workers=2)
        assert result == []

    def test_empty_list_with_accumulator(self) -> None:
        result = process(
            [],
            double,
            num_workers=2,
            accumulator_fn=lambda acc, x: acc + x,
            initial_value=42,
        )
        assert result == 42

    def test_single_item(self) -> None:
        result = process([5], square, num_workers=1)
        assert result == [25]

    def test_fail_fast(self) -> None:
        with pytest.raises(TaskFailedError) as exc_info:
            process([1, 2, 3], fail_always, num_workers=2)
        assert exc_info.value.stats.failed >= 1
        assert isinstance(exc_info.value.original_exception, ValueError)

    def test_retry_succeeds(self) -> None:
        result = process([100], fail_once_then_succeed, num_workers=1, max_retries=2)
        assert result == [200]

    def test_retry_exhausted(self) -> None:
        with pytest.raises(TaskFailedError):
            process([1], fail_always, num_workers=1, max_retries=2)

    def test_num_workers(self) -> None:
        result = process([1, 2, 3, 4, 5], slow_double, num_workers=4)
        assert sorted(result) == [2, 4, 6, 8, 10]

    def test_accumulator_collect_to_set(self) -> None:
        result = process(
            [1, 2, 3, 2, 1],
            double,
            num_workers=2,
            accumulator_fn=lambda acc, x: acc | {x},
            initial_value=set(),
        )
        assert result == {2, 4, 6}
