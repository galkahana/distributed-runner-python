# Distributed Runner

A Python library for distributing work across processes with automatic retry and optional result accumulation. Supports batch processing, simplified map-reduce, and continuous streaming patterns. Uses `ProcessPoolExecutor` from `concurrent.futures`.

## Installation

```bash
pip install git+https://github.com/galkahana/distributed-runner-python.git@v1.0.0
```

## Use Cases

- [**Batch Processing**](#1-batch-processing) — Process a list of items in parallel, optionally accumulate results
- [**Partitioned Parallel Processing**](#2-partitioned-parallel-processing) — Split data into partitions, process each in parallel
- [**Continuous Feed Processing**](#3-continuous-feed-processing) — Accumulator-driven workflow that can submit new tasks and control shutdown

### 1. Batch Processing

Use `process()` to process a list of items in parallel. Without an accumulator, results are discarded and `None` is returned — useful for fire-and-forget or side-effect workloads:

```python
from distributed_runner import process

def double(x: int) -> int:
    return x * 2

# Without accumulator — results are discarded, returns None
process([1, 2, 3, 4, 5], double, num_workers=4)
```

> **Note:** Task functions must be top-level or static — they are sent to worker processes via pickling. Lambdas and closures are not picklable and will fail at runtime.

With an accumulator, results are folded into a single value:

```python
total = process(
    [1, 2, 3, 4, 5],
    double,
    num_workers=4,
    accumulator_fn=lambda acc, x: acc + x,
    initial_value=0,
)
# total = 30
```

Use `on_progress` to track progress as tasks complete. The callback receives a `Stats` snapshot after each task:

```python
from distributed_runner import Stats

def report(stats: Stats) -> None:
    print(f"{stats.completed}/{stats.submitted} complete, {stats.failed} failed")

total = process(
    [1, 2, 3, 4, 5],
    double,
    num_workers=4,
    accumulator_fn=lambda acc, x: acc + x,
    initial_value=0,
    on_progress=report,
)
```

### 2. Partitioned Parallel Processing

Use `map_reduce()` to split work into partitions processed by a process pool. Each partition receives its index, total count, and shared data — enough to independently determine what slice of work to do.

**Example: export rows from a database to Kafka**

Each partition receives the full date range and computes its own sub-range, fetches from the database, and publishes to Kafka. No accumulation needed — returns `None`.

```python
from datetime import date, timedelta
from distributed_runner import map_reduce, PartitionTask

def export_partition(task: PartitionTask[tuple[date, date]]) -> None:
    lookback_date, end_date = task.shared_data
    total_days = (end_date - lookback_date).days
    days_per_partition = total_days // task.total_partitions

    start = lookback_date + timedelta(days=task.partition_index * days_per_partition)
    end = (
        end_date
        if task.partition_index == task.total_partitions - 1
        else start + timedelta(days=days_per_partition)
    )

    rows = db.fetch_by_date_range(start, end)  # your DB call
    for row in rows:
        kafka.publish("export-topic", row)  # your Kafka publish

map_reduce(
    (date(2023, 1, 1), date(2024, 1, 1)),   # (lookback_date, end_date), shared across all partitions
    num_partitions=8,
    process_fn=export_partition,
    num_workers=4,
)  # returns None
```

**Example: compute statistics over transaction amounts**

Each partition processes its slice and returns partial stats. The accumulator merges them into a final result.

```python
from dataclasses import dataclass

@dataclass
class PartitionStats:
    total: float
    count: int
    minimum: float
    maximum: float

def compute_stats(task: PartitionTask[list[float]]) -> PartitionStats:
    amounts = task.shared_data
    chunk_size = len(amounts) // task.total_partitions
    start = task.partition_index * chunk_size
    end = len(amounts) if task.partition_index == task.total_partitions - 1 else start + chunk_size

    chunk = amounts[start:end]
    return PartitionStats(
        total=sum(chunk),
        count=len(chunk),
        minimum=min(chunk),
        maximum=max(chunk),
    )

def merge_stats(acc: PartitionStats, stats: PartitionStats) -> PartitionStats:
    return PartitionStats(
        total=acc.total + stats.total,
        count=acc.count + stats.count,
        minimum=min(acc.minimum, stats.minimum),
        maximum=max(acc.maximum, stats.maximum),
    )

result = map_reduce(
    transaction_amounts,    # list[float], shared across all partitions
    num_partitions=8,
    process_fn=compute_stats,
    num_workers=4,
    accumulator_fn=merge_stats,
    initial_value=PartitionStats(total=0, count=0, minimum=float("inf"), maximum=float("-inf")),
)
average = result.total / result.count
```

### 3. Continuous Feed Processing

Use `process_continuous()` for accumulator-driven workflows. The accumulator receives each result, can submit new tasks via `submit_fn`, and controls shutdown by returning `(new_acc, should_continue)`.

```python
from distributed_runner import process_continuous

def crawl(url: str) -> dict:
    """Fetch a URL and extract links."""
    # ... fetch and parse ...
    return {"url": url, "links": found_links, "content": page_content}

def accumulate(acc, result, submit):
    visited = acc["visited"] | {result["url"]}
    pages = {**acc["pages"], result["url"]: result["content"]}

    # Submit newly discovered links
    for link in result["links"]:
        if link not in visited:
            submit(link)

    should_continue = len(visited) < 1000  # stop after 1000 pages
    return ({"visited": visited, "pages": pages}, should_continue)

result = process_continuous(
    ["https://example.com"],
    crawl,
    accumulate,
    initial_value={"visited": set(), "pages": {}},
    num_workers=8,
)
```

## Features

- **Fail-fast error handling** — On first failure, remaining tasks are cancelled and `TaskFailedError` is raised with stats
- **Automatic retry** — Simple retry with no external dependencies, runs inside worker processes (no IPC overhead)
- **Progress callbacks** — `on_progress` callback invoked after each task completes with current `Stats`, useful for progress bars and logging
- **Stats tracking** — `TaskFailedError.stats` and `on_progress` both report submitted, completed, and failed counts

## Development

Requires Python 3.10+.

```bash
make install        # install dependencies
make check          # runs: ruff format --check, ruff check, mypy --strict, pytest
make test           # just tests
make format         # auto-format
```
