"""Wave-461 DAG-2/shifted-symplectic bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.derived_critical import bench_derived_critical
from quant_fund.models.lagrangian_int import bench_lagrangian_int
from quant_fund.models.lie_algebroid import bench_lie_algebroid
from quant_fund.models.moment_map import bench_moment_map
from quant_fund.models.quant_dag import bench_quant_dag
from quant_fund.models.shifted_sympl import bench_shifted_sympl

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


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


def bench_shifted_sympl_family(seed: int = _SEED + 2672) -> dict[str, float]:
    return _floats(_finite_blob("shifted_sympl", bench_shifted_sympl(seed)))


def bench_lagrangian_int_family(seed: int = _SEED + 2673) -> dict[str, float]:
    return _floats(_finite_blob("lagrangian_int", bench_lagrangian_int(seed)))


def bench_derived_critical_family(seed: int = _SEED + 2674) -> dict[str, float]:
    return _floats(_finite_blob("derived_critical", bench_derived_critical(seed)))


def bench_lie_algebroid_family(seed: int = _SEED + 2675) -> dict[str, float]:
    return _floats(_finite_blob("lie_algebroid", bench_lie_algebroid(seed)))


def bench_moment_map_family(seed: int = _SEED + 2676) -> dict[str, float]:
    return _floats(_finite_blob("moment_map", bench_moment_map(seed)))


def bench_quant_dag_family(seed: int = _SEED + 2677) -> dict[str, float]:
    return _floats(_finite_blob("quant_dag", bench_quant_dag(seed)))
