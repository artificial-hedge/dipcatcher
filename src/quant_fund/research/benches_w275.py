"""Wave-275 databases-4 benches: execution internals."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.columnar_scan import bench_columnar_scan
from quant_fund.models.graceful_hash import bench_graceful_hash
from quant_fund.models.index_intersect import bench_index_intersect
from quant_fund.models.late_materialize import bench_late_materialize
from quant_fund.models.radix_join import bench_radix_join
from quant_fund.models.simd_filter import bench_simd_filter

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


def bench_columnar_scan_family(seed: int = _SEED + 1520) -> dict[str, float]:
    return _floats(_finite_blob("columnar_scan", bench_columnar_scan(seed)))


def bench_simd_filter_family(seed: int = _SEED + 1521) -> dict[str, float]:
    return _floats(_finite_blob("simd_filter", bench_simd_filter(seed)))


def bench_late_materialize_family(seed: int = _SEED + 1522) -> dict[str, float]:
    return _floats(_finite_blob("late_materialize", bench_late_materialize(seed)))


def bench_radix_join_family(seed: int = _SEED + 1523) -> dict[str, float]:
    return _floats(_finite_blob("radix_join", bench_radix_join(seed)))


def bench_graceful_hash_family(seed: int = _SEED + 1524) -> dict[str, float]:
    return _floats(_finite_blob("graceful_hash", bench_graceful_hash(seed)))


def bench_index_intersect_family(seed: int = _SEED + 1525) -> dict[str, float]:
    return _floats(_finite_blob("index_intersect", bench_index_intersect(seed)))
