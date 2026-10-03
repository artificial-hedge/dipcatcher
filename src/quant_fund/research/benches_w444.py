"""Wave-444 DAG-stacks bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cotangent_cx import bench_cotangent_cx
from quant_fund.models.derived_stack import bench_derived_stack
from quant_fund.models.geometric_stk import bench_geometric_stk
from quant_fund.models.perf_stack import bench_perf_stack
from quant_fund.models.quasi_smooth import bench_quasi_smooth
from quant_fund.models.tannaka_rec import bench_tannaka_rec

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


def bench_derived_stack_family(seed: int = _SEED + 2570) -> dict[str, float]:
    return _floats(_finite_blob("derived_stack", bench_derived_stack(seed)))


def bench_cotangent_cx_family(seed: int = _SEED + 2571) -> dict[str, float]:
    return _floats(_finite_blob("cotangent_cx", bench_cotangent_cx(seed)))


def bench_geometric_stk_family(seed: int = _SEED + 2572) -> dict[str, float]:
    return _floats(_finite_blob("geometric_stk", bench_geometric_stk(seed)))


def bench_tannaka_rec_family(seed: int = _SEED + 2573) -> dict[str, float]:
    return _floats(_finite_blob("tannaka_rec", bench_tannaka_rec(seed)))


def bench_quasi_smooth_family(seed: int = _SEED + 2574) -> dict[str, float]:
    return _floats(_finite_blob("quasi_smooth", bench_quasi_smooth(seed)))


def bench_perf_stack_family(seed: int = _SEED + 2575) -> dict[str, float]:
    return _floats(_finite_blob("perf_stack", bench_perf_stack(seed)))
