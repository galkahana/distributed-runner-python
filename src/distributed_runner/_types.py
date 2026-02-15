from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

I = TypeVar("I")
O = TypeVar("O")
A = TypeVar("A")
D = TypeVar("D")


@dataclass(frozen=True)
class PartitionTask(Generic[D]):
    partition_index: int
    total_partitions: int
    shared_data: D
