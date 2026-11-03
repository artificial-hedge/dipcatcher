#!/usr/bin/env python3
"""One-hour simulated paper soak. RSS, file descriptors, and threads.

Each cycle opens a book, submits one order, and deletes that directory.
Samples are the soak process itself. After the warmup, a file-descriptor
slope above 0.05 per second, an RSS slope above 20 KiB per second, or a
thread count that grows by more than 1 fails the run. Stdout is one JSON
object and the same object is written to ``DIPCATCHER_PAPER_SOAK_JSON``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

from quant_fund.paper.durable_book import open_book, submit_once

_DEFAULT_SECONDS = 3600.0
_WARMUP_S = 120.0
_INTERVAL_S = 30.0
_FD_PER_S = 0.05
_RSS_KB_PER_S = 20.0
_THREAD_GROWTH = 1
_WORK = Path("/tmp/dipcatcher-ops16-soak-work")


def slope(points: list[tuple[float, float]]) -> float:
    """Least-squares slope of ``y`` against ``x``. Zero when undefined."""
    count = len(points)
    if count < 2:
        return 0.0
    mean_x = sum(point[0] for point in points) / count
    mean_y = sum(point[1] for point in points) / count
    numerator = sum((point[0] - mean_x) * (point[1] - mean_y) for point in points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def leak_bounds(
    samples: list[dict[str, float]],
    *,
    warmup_s: float = _WARMUP_S,
) -> dict[str, object]:
    """Flag an RSS climb, a descriptor climb, or a thread climb after warmup."""
    if not samples:
        return {
            "fd_per_s": 0.0,
            "leak": True,
            "rss_kb_per_s": 0.0,
            "samples": 0,
            "thread_growth": 0,
            "thread_max": 0,
        }
    origin = samples[0]["t"]
    kept = [item for item in samples if item["t"] - origin >= warmup_s]
    if len(kept) < 3:
        kept = samples
    fd_slope = slope([(item["t"], item["fds"]) for item in kept])
    rss_slope = slope([(item["t"], item["rss_kb"]) for item in kept])
    threads = [int(item["threads"]) for item in kept]
    growth = max(threads) - min(threads)
    leak = fd_slope > _FD_PER_S or rss_slope > _RSS_KB_PER_S or growth > _THREAD_GROWTH
    return {
        "fd_per_s": fd_slope,
        "leak": leak,
        "rss_kb_per_s": rss_slope,
        "samples": len(kept),
        "thread_growth": growth,
        "thread_max": max(threads),
    }


def _sample(elapsed: float) -> dict[str, float]:
    rss_kb = 0
    threads = 0
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            rss_kb = int(line.split()[1])
        elif line.startswith("Threads:"):
            threads = int(line.split()[1])
    fds = sum(1 for _entry in os.scandir("/proc/self/fd"))
    return {"fds": float(fds), "rss_kb": float(rss_kb), "t": elapsed, "threads": float(threads)}


def _cycle(index: int) -> None:
    state = _WORK / "book"
    if state.exists():
        shutil.rmtree(state)
    session = open_book(state)
    verdict = submit_once(session, state, f"soak-{index}")
    if verdict != "accepted":
        raise RuntimeError(verdict)
    del session


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value < 0:
        return default
    return value


def execute(seconds: float, interval: float, warmup: float) -> dict[str, object]:
    if _WORK.exists():
        shutil.rmtree(_WORK)
    _WORK.mkdir(parents=True)
    samples: list[dict[str, float]] = []
    started = time.monotonic()
    next_sample = 0.0
    cycles = 0
    while True:
        elapsed = time.monotonic() - started
        if elapsed >= seconds:
            break
        _cycle(cycles)
        cycles += 1
        elapsed = time.monotonic() - started
        if elapsed >= next_sample:
            samples.append(_sample(elapsed))
            next_sample += interval
    decision = leak_bounds(samples, warmup_s=warmup)
    report: dict[str, object] = {
        "cycles": cycles,
        "error": "" if not decision["leak"] else "leak",
        "interval_s": interval,
        "ok": not bool(decision["leak"]),
        "seconds": seconds,
        "series": samples,
        "warmup_s": warmup,
    }
    report.update(decision)
    path = Path(os.environ.get("DIPCATCHER_PAPER_SOAK_JSON", "/tmp/ops16-paper-soak.json"))
    path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.stderr.write("prod-paper-soak takes no arguments\n")
        return 1
    seconds = _float_env("DIPCATCHER_PAPER_SOAK_SECONDS", _DEFAULT_SECONDS)
    interval = _float_env("DIPCATCHER_PAPER_SOAK_INTERVAL", _INTERVAL_S)
    warmup = _float_env("DIPCATCHER_PAPER_SOAK_WARMUP", _WARMUP_S)
    if interval <= 0:
        interval = _INTERVAL_S
    report = execute(seconds, interval, warmup)
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
