from __future__ import annotations

import pytest

from distributed_runner import TaskFailedError, map_reduce
from tests.helpers import partition_fail, partition_sum


class TestMapReduce:
    def test_basic_partition_sum(self) -> None:
        data = list(range(1, 101))  # 1..100
        result = map_reduce(
            data,
            num_partitions=4,
            process_fn=partition_sum,
            num_workers=2,
            accumulator_fn=lambda acc, x: acc + x,
            initial_value=0,
        )
        assert result == sum(range(1, 101))

    def test_no_accumulator(self) -> None:
        data = list(range(1, 11))
        result = map_reduce(
            data,
            num_partitions=2,
            process_fn=partition_sum,
            num_workers=2,
        )
        assert result is None

    def test_single_partition(self) -> None:
        data = [10, 20, 30]
        result = map_reduce(
            data,
            num_partitions=1,
            process_fn=partition_sum,
            num_workers=1,
            accumulator_fn=lambda acc, x: acc + x,
            initial_value=0,
        )
        assert result == 60

    def test_partition_failure(self) -> None:
        with pytest.raises(TaskFailedError) as exc_info:
            map_reduce(
                [1, 2, 3],
                num_partitions=2,
                process_fn=partition_fail,
                num_workers=2,
            )
        assert isinstance(exc_info.value.original_exception, ValueError)
