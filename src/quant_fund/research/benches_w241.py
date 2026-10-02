"""Wave-241 adapters: databases-2 canon — ARIES recovery, strict 2PL,
Selinger join DP, MVCC GC, buffer pool, B-link tree — SYNTHETIC
correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.aries_recovery import bench_aries_recovery
from quant_fund.models.blink_tree import bench_blink_tree
from quant_fund.models.buffer_pool import bench_buffer_pool
from quant_fund.models.mvcc_gc import bench_mvcc_gc
from quant_fund.models.selinger_join import bench_selinger_join
from quant_fund.models.two_phase_lock import bench_two_phase_lock

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


def bench_aries_recovery_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("aries_recovery", bench_aries_recovery(seed=_SEED + 1180)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"aries_recovery bench failed: {exc}") from exc


def bench_blink_tree_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("blink_tree", bench_blink_tree(seed=_SEED + 1181)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"blink_tree bench failed: {exc}") from exc


def bench_buffer_pool_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("buffer_pool", bench_buffer_pool(seed=_SEED + 1182)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"buffer_pool bench failed: {exc}") from exc


def bench_mvcc_gc_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mvcc_gc", bench_mvcc_gc(seed=_SEED + 1183)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mvcc_gc bench failed: {exc}") from exc


def bench_selinger_join_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("selinger_join", bench_selinger_join(seed=_SEED + 1184)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"selinger_join bench failed: {exc}") from exc


def bench_two_phase_lock_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("two_phase_lock", bench_two_phase_lock(seed=_SEED + 1185)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"two_phase_lock bench failed: {exc}") from exc
