from __future__ import annotations

from distributed_runner._stats import Stats


class DistributedRunnerError(Exception):
    pass


class TaskFailedError(DistributedRunnerError):
    def __init__(self, message: str, original_exception: BaseException, stats: Stats) -> None:
        super().__init__(message)
        self.original_exception = original_exception
        self.stats = stats
