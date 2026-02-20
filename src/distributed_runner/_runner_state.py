from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from distributed_runner._errors import TaskFailedError
from distributed_runner._stats import Stats


@dataclass
class RunnerState:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    accumulator: Any = None

    def to_stats(self) -> Stats:
        return Stats(submitted=self.submitted, completed=self.completed, failed=self.failed)

    def get_result_or_fail(self, future: Any, pending: Iterable[Any]) -> Any:
        try:
            return future.result()
        except Exception as exc:
            self.failed += 1
            for f in pending:
                f.cancel()
            raise TaskFailedError(
                f"Task failed: {exc}",
                original_exception=exc,
                stats=self.to_stats(),
            ) from exc
