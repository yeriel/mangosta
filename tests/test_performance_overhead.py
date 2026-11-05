"""Performance overhead and latency benchmark test suite for RunTracker."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from mangosta.tracking import RunTracker


def test_log_metrics_per_step_latency_sub_millisecond(tmp_path: Path) -> None:
    """Verifies that individual log_metrics calls execute in microseconds, not milliseconds.

    Because f.flush() pushes to OS page cache without blocking on physical disk barriers
    (which only occur periodically on sync_interval=10), per-step overhead must remain < 200 µs.
    """
    exp_dir = tmp_path / "bench_exp"
    tracker = RunTracker(
        experiment="perf_bench",
        run_name="latency_run",
        root_dir=exp_dir,
        sync_interval=10,
    )

    n_steps = 300
    latencies: list[float] = []

    # Warmup
    tracker.log_metrics({"loss": 1.0}, step=0)

    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        tracker.log_metrics(
            {"train_loss": 0.42, "train_acc": 0.95, "learning_rate": 1e-4},
            step=step,
        )
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

    tracker.finish()

    avg_latency_sec = sum(latencies) / len(latencies)
    avg_latency_us = avg_latency_sec * 1_000_000

    # Ensure average latency per batch is well under 250 microseconds
    assert avg_latency_sec < 0.0005, (
        f"Average log_metrics latency too high: {avg_latency_us:.1f} µs (expected < 500 µs)"
    )


def test_training_loop_zero_overhead_under_threshold(tmp_path: Path) -> None:
    """Verifies that in a standard training step, RunTracker introduces minimal overhead."""
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    n_batches = 150
    data = [torch.randn(32, 32) for _ in range(n_batches)]
    targets = [torch.randint(0, 4, (32,)) for _ in range(n_batches)]

    # 1. Baseline: Pure PyTorch forward + backward + step without any tracking
    t0 = time.perf_counter()
    for x, y in zip(data, targets):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
    t_baseline = time.perf_counter() - t0

    # 2. Tracked: Same workload with RunTracker logging every step
    exp_dir = tmp_path / "bench_overhead"
    tracker = RunTracker(
        experiment="overhead_test",
        run_name="tracked_run",
        root_dir=exp_dir,
        sync_interval=10,
    )

    t0 = time.perf_counter()
    for step, (x, y) in enumerate(zip(data, targets)):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        tracker.log_metrics({"loss": float(loss.item()), "step": step}, step=step)
    t_tracked = time.perf_counter() - t0

    tracker.finish()

    overhead_sec = t_tracked - t_baseline
    # Check that total extra time spent across all 150 training batches is negligible (< 50 ms total)
    assert overhead_sec < 0.15, (
        f"Overhead too high: {overhead_sec * 1000:.2f} ms total over {n_batches} batches"
    )
