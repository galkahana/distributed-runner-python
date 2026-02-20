"""distributed-runner: Distribute work across multiple processes."""

from distributed_runner._batch import process
from distributed_runner._continuous import process_continuous
from distributed_runner._errors import DistributedRunnerError, TaskFailedError
from distributed_runner._map_reduce import map_reduce
from distributed_runner._stats import Stats
from distributed_runner._types import PartitionTask

__all__ = [
    "map_reduce",
    "process",
    "process_continuous",
    "DistributedRunnerError",
    "TaskFailedError",
    "PartitionTask",
    "Stats",
]
