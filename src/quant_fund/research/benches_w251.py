"""Wave-251 adapters: numerical-linalg-2 canon — LDLᵀ + rank-1
update, Givens QR, one-sided Jacobi SVD, orthogonal iteration,
LU partial pivoting, Sturm bisection — SYNTHETIC benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.givens_qr import bench_givens_qr
from quant_fund.models.jacobi_svd import bench_jacobi_svd
from quant_fund.models.ldlt_solve import bench_ldlt_solve
from quant_fund.models.lu_pivots import bench_lu_pivots
from quant_fund.models.orth_iter import bench_orth_iter
from quant_fund.models.sturm_eig import bench_sturm_eig

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


def bench_givens_qr_family(seed: int = _SEED + 1280) -> dict[str, float]:
    return bench_givens_qr(seed)


def bench_jacobi_svd_family(seed: int = _SEED + 1281) -> dict[str, float]:
    return bench_jacobi_svd(seed)


def bench_ldlt_solve_family(seed: int = _SEED + 1282) -> dict[str, float]:
    return bench_ldlt_solve(seed)


def bench_lu_pivots_family(seed: int = _SEED + 1283) -> dict[str, float]:
    return bench_lu_pivots(seed)


def bench_orth_iter_family(seed: int = _SEED + 1284) -> dict[str, float]:
    return bench_orth_iter(seed)


def bench_sturm_eig_family(seed: int = _SEED + 1285) -> dict[str, float]:
    return bench_sturm_eig(seed)
