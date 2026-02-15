from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stats:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
