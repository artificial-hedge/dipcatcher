"""Wave-460 condensed-2/analytic-rings bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.analytic_ring2 import bench_analytic_ring2
from quant_fund.models.clausen_scholze import bench_clausen_scholze
from quant_fund.models.pyknotic import bench_pyknotic
from quant_fund.models.solid_derived import bench_solid_derived
from quant_fund.models.solid_tensor import bench_solid_tensor
from quant_fund.models.trace_class import bench_trace_class

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


def bench_analytic_ring2_family(seed: int = _SEED + 2666) -> dict[str, float]:
    return _floats(_finite_blob("analytic_ring2", bench_analytic_ring2(seed)))


def bench_solid_tensor_family(seed: int = _SEED + 2667) -> dict[str, float]:
    return _floats(_finite_blob("solid_tensor", bench_solid_tensor(seed)))


def bench_trace_class_family(seed: int = _SEED + 2668) -> dict[str, float]:
    return _floats(_finite_blob("trace_class", bench_trace_class(seed)))


def bench_clausen_scholze_family(seed: int = _SEED + 2669) -> dict[str, float]:
    return _floats(_finite_blob("clausen_scholze", bench_clausen_scholze(seed)))


def bench_solid_derived_family(seed: int = _SEED + 2670) -> dict[str, float]:
    return _floats(_finite_blob("solid_derived", bench_solid_derived(seed)))


def bench_pyknotic_family(seed: int = _SEED + 2671) -> dict[str, float]:
    return _floats(_finite_blob("pyknotic", bench_pyknotic(seed)))
