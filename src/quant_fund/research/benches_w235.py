"""Wave-235 adapters: OS-internals canon — RR/CFS scheduling, demand
paging, deadlock + Banker's, disk scheduling, journaled FS — SYNTHETIC
correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cfs_scheduler import bench_cfs_scheduler
from quant_fund.models.deadlock_detect import bench_deadlock_detect
from quant_fund.models.demand_paging import bench_demand_paging
from quant_fund.models.disk_sched import bench_disk_sched
from quant_fund.models.fs_journal import bench_fs_journal
from quant_fund.models.round_robin_sched import bench_round_robin_sched

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_cfs_scheduler_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cfs_scheduler", bench_cfs_scheduler(seed=_SEED + 1120)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cfs_scheduler bench failed: {exc}") from exc


def bench_deadlock_detect_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("deadlock_detect", bench_deadlock_detect(seed=_SEED + 1121)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"deadlock_detect bench failed: {exc}") from exc


def bench_demand_paging_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("demand_paging", bench_demand_paging(seed=_SEED + 1122)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"demand_paging bench failed: {exc}") from exc


def bench_disk_sched_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("disk_sched", bench_disk_sched(seed=_SEED + 1123)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"disk_sched bench failed: {exc}") from exc


def bench_fs_journal_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fs_journal", bench_fs_journal(seed=_SEED + 1124)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fs_journal bench failed: {exc}") from exc


def bench_round_robin_sched_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("round_robin_sched", bench_round_robin_sched(seed=_SEED + 1125))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"round_robin_sched bench failed: {exc}") from exc
