"""Wave-458 matroid-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.delta_matroid import bench_delta_matroid
from quant_fund.models.matroid_minor import bench_matroid_minor
from quant_fund.models.matroid_rep import bench_matroid_rep
from quant_fund.models.regular_mat import bench_regular_mat
from quant_fund.models.transversal_mat import bench_transversal_mat
from quant_fund.models.tutte_poly import bench_tutte_poly

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


def bench_transversal_mat_family(seed: int = _SEED + 2654) -> dict[str, float]:
    return _floats(_finite_blob("transversal_mat", bench_transversal_mat(seed)))


def bench_matroid_rep_family(seed: int = _SEED + 2655) -> dict[str, float]:
    return _floats(_finite_blob("matroid_rep", bench_matroid_rep(seed)))


def bench_tutte_poly_family(seed: int = _SEED + 2656) -> dict[str, float]:
    return _floats(_finite_blob("tutte_poly", bench_tutte_poly(seed)))


def bench_matroid_minor_family(seed: int = _SEED + 2657) -> dict[str, float]:
    return _floats(_finite_blob("matroid_minor", bench_matroid_minor(seed)))


def bench_regular_mat_family(seed: int = _SEED + 2658) -> dict[str, float]:
    return _floats(_finite_blob("regular_mat", bench_regular_mat(seed)))


def bench_delta_matroid_family(seed: int = _SEED + 2659) -> dict[str, float]:
    return _floats(_finite_blob("delta_matroid", bench_delta_matroid(seed)))
