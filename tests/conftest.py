from __future__ import annotations

import pytest

from tests.helpers import FailOnceThenSucceed, PartitionFailOnceThenSucceed


@pytest.fixture
def fail_once_then_succeed() -> FailOnceThenSucceed:
    return FailOnceThenSucceed()


@pytest.fixture
def partition_fail_once_then_succeed() -> PartitionFailOnceThenSucceed:
    return PartitionFailOnceThenSucceed()
