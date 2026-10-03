"""Wave-255 adapters: optimization-2 canon — simplex tableau,
ellipsoid method, log-barrier IPM, ADMM lasso, coordinate
descent, projected gradient — SYNTHETIC benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.admm_lasso import bench_admm_lasso
from quant_fund.models.barrier_ip import bench_barrier_ip
from quant_fund.models.coord_descent import bench_coord_descent
from quant_fund.models.ellipsoid_method import bench_ellipsoid_method
from quant_fund.models.proj_gradient import bench_proj_gradient
from quant_fund.models.simplex_lp import bench_simplex_lp

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


def bench_simplex_lp_family(seed: int = _SEED + 1320) -> dict[str, float]:
    return bench_simplex_lp(seed)


def bench_ellipsoid_method_family(seed: int = _SEED + 1321) -> dict[str, float]:
    return bench_ellipsoid_method(seed)


def bench_barrier_ip_family(seed: int = _SEED + 1322) -> dict[str, float]:
    return bench_barrier_ip(seed)


def bench_admm_lasso_family(seed: int = _SEED + 1323) -> dict[str, float]:
    return bench_admm_lasso(seed)


def bench_coord_descent_family(seed: int = _SEED + 1324) -> dict[str, float]:
    return bench_coord_descent(seed)


def bench_proj_gradient_family(seed: int = _SEED + 1325) -> dict[str, float]:
    return bench_proj_gradient(seed)
