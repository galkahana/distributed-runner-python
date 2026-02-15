# Distributed Runner

A Python library for distributing work across processes with automatic retry and optional result accumulation. Supports batch processing, simplified map-reduce, and continuous streaming patterns. Uses `ProcessPoolExecutor` from `concurrent.futures`.

## Installation

```bash
pip install distributed-runner
```

Or with Poetry:

```bash
poetry add distributed-runner
```

## Use Cases

- [**Batch Processing with Accumulation**](#1-batch-processing-with-accumulation) — Process a list of items in parallel, accumulate results into a single value
- [**Partitioned Parallel Processing**](#2-partitioned-parallel-processing) — Split data into partitions, process each in parallel
- [**Continuous Feed Processing**](#3-continuous-feed-processing) — Accumulator-driven workflow that can submit new tasks and control shutdown

### 1. Batch Processing with Accumulation

Use `process()` to process a list of items in parallel and accumulate results into a single value. Without an accumulator, returns a list of results.

```python
from distributed_runner import process

# Simple parallel processing — returns a list of results
results = process([1, 2, 3, 4, 5], lambda x: x * 2, num_workers=4)
# results = [2, 4, 6, 8, 10]  (order not guaranteed)
```

```python
# With accumulation — fold results into a single value
total = process(
    [1, 2, 3, 4, 5],
    lambda x: x * 2,
    num_workers=4,
    accumulator_fn=lambda acc, x: acc + x,
    initial_value=0,
)
# total = 30
```

Async version:

```python
from distributed_runner import async_process

total = await async_process(
    [1, 2, 3, 4, 5],
    lambda x: x * 2,
    num_workers=4,
    accumulator_fn=lambda acc, x: acc + x,
    initial_value=0,
)
```

### 2. Partitioned Parallel Processing

Use `map_reduce()` to split work into partitions processed by a process pool. Each partition receives its index and total count, allowing it to select its slice of data independently.

This example reads database rows using range-based partitioning and writes each row to a file:

```python
from distributed_runner import map_reduce, PartitionTask

def process_partition(task: PartitionTask[list[int]]) -> int:
    """Sum elements in this partition's slice of the shared data."""
    data = task.shared_data
    chunk_size = len(data) // task.total_partitions
    start = task.partition_index * chunk_size
    end = len(data) if task.partition_index == task.total_partitions - 1 else start + chunk_size
    return sum(data[start:end])

data = list(range(1, 10_001))
total = map_reduce(
    data,
    num_partitions=8,
    process_fn=process_partition,
    num_workers=4,
    accumulator_fn=lambda acc, x: acc + x,
    initial_value=0,
)
# total = 50_005_000
```

Async version:

```python
from distributed_runner import async_map_reduce

total = await async_map_reduce(
    data,
    num_partitions=8,
    process_fn=process_partition,
    num_workers=4,
    accumulator_fn=lambda acc, x: acc + x,
    initial_value=0,
)
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

Async version:

```python
from distributed_runner import async_process_continuous

result = await async_process_continuous(
    ["https://example.com"],
    crawl,
    accumulate,
    initial_value={"visited": set(), "pages": {}},
    num_workers=8,
)
```

## Features

- **Fail-fast error handling** — On first failure, remaining tasks are cancelled and `TaskFailedError` is raised with stats
- **Automatic retry** — Tenacity-based retry with exponential backoff, runs inside worker processes (no IPC overhead)
- **Stats tracking** — `TaskFailedError.stats` reports submitted, completed, and failed counts
- **Both sync and async** — Every function has an `async_` counterpart

## Development

Requires Python 3.10+.

```bash
make install        # install dependencies
make check          # runs: ruff format --check, ruff check, mypy --strict, pytest
make test           # just tests
make format         # auto-format
```
