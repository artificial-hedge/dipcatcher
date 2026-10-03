"""Wave-385 commutative-algebra-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.completion_ring import bench_completion_ring
from quant_fund.models.dimension_fiber import bench_dimension_fiber
from quant_fund.models.hilbert_samuel import bench_hilbert_samuel
from quant_fund.models.krull_dim import bench_krull_dim
from quant_fund.models.noether_normal import bench_noether_normal
from quant_fund.models.primary_decomp import bench_primary_decomp

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


def bench_hilbert_samuel_family(seed: int = _SEED + 2216) -> dict[str, float]:
    return _floats(_finite_blob("hilbert_samuel", bench_hilbert_samuel(seed)))


def bench_krull_dim_family(seed: int = _SEED + 2217) -> dict[str, float]:
    return _floats(_finite_blob("krull_dim", bench_krull_dim(seed)))


def bench_noether_normal_family(seed: int = _SEED + 2218) -> dict[str, float]:
    return _floats(_finite_blob("noether_normal", bench_noether_normal(seed)))


def bench_primary_decomp_family(seed: int = _SEED + 2219) -> dict[str, float]:
    return _floats(_finite_blob("primary_decomp", bench_primary_decomp(seed)))


def bench_completion_ring_family(seed: int = _SEED + 2220) -> dict[str, float]:
    return _floats(_finite_blob("completion_ring", bench_completion_ring(seed)))


def bench_dimension_fiber_family(seed: int = _SEED + 2221) -> dict[str, float]:
    return _floats(_finite_blob("dimension_fiber", bench_dimension_fiber(seed)))
