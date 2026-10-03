"""Wave-247 adapters: concurrency canon — Peterson/bakery locks,
RW lock, TAS/CAS atomics, work stealing, CSP select — SYNTHETIC
correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.atomics_tas import bench_atomics_tas
from quant_fund.models.bakery_lock import bench_bakery_lock
from quant_fund.models.channel_select import bench_channel_select
from quant_fund.models.peterson_lock import bench_peterson_lock
from quant_fund.models.rw_lock import bench_rw_lock
from quant_fund.models.work_stealing import bench_work_stealing

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


def bench_atomics_tas_family(seed: int = _SEED + 1240) -> dict[str, float]:
    return bench_atomics_tas(seed)


def bench_bakery_lock_family(seed: int = _SEED + 1241) -> dict[str, float]:
    return bench_bakery_lock(seed)


def bench_channel_select_family(seed: int = _SEED + 1242) -> dict[str, float]:
    return bench_channel_select(seed)


def bench_peterson_lock_family(seed: int = _SEED + 1243) -> dict[str, float]:
    return bench_peterson_lock(seed)


def bench_rw_lock_family(seed: int = _SEED + 1244) -> dict[str, float]:
    return bench_rw_lock(seed)


def bench_work_stealing_family(seed: int = _SEED + 1245) -> dict[str, float]:
    return bench_work_stealing(seed)
