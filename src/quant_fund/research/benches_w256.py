"""Wave-256 adapters: memory-models canon — hazard pointers,
seqlock, Michael-Scott queue, epoch-based reclamation,
flat combining, RCU — SYNTHETIC benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.epoch_reclaim import bench_epoch_reclaim
from quant_fund.models.flat_combining import bench_flat_combining
from quant_fund.models.hazard_pointer import bench_hazard_pointer
from quant_fund.models.ms_queue import bench_ms_queue
from quant_fund.models.rcu_lock import bench_rcu_lock
from quant_fund.models.seqlock import bench_seqlock

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


def bench_hazard_pointer_family(seed: int = _SEED + 1330) -> dict[str, float]:
    return bench_hazard_pointer(seed)


def bench_seqlock_family(seed: int = _SEED + 1331) -> dict[str, float]:
    return bench_seqlock(seed)


def bench_ms_queue_family(seed: int = _SEED + 1332) -> dict[str, float]:
    return bench_ms_queue(seed)


def bench_epoch_reclaim_family(seed: int = _SEED + 1333) -> dict[str, float]:
    return bench_epoch_reclaim(seed)


def bench_flat_combining_family(seed: int = _SEED + 1334) -> dict[str, float]:
    return bench_flat_combining(seed)


def bench_rcu_lock_family(seed: int = _SEED + 1335) -> dict[str, float]:
    return bench_rcu_lock(seed)
