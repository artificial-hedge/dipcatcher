"""Wave-333 computer-algebra-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.hensel_lift import bench_hensel_lift
from quant_fund.models.poly_crt import bench_poly_crt
from quant_fund.models.poly_eval_interp import bench_poly_eval_interp
from quant_fund.models.poly_factor_fp import bench_poly_factor_fp
from quant_fund.models.sparse_interp import bench_sparse_interp
from quant_fund.models.subresultant import bench_subresultant

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


def bench_poly_factor_fp_family(seed: int = _SEED + 1905) -> dict[str, float]:
    return _floats(_finite_blob("poly_factor_fp", bench_poly_factor_fp(seed)))


def bench_hensel_lift_family(seed: int = _SEED + 1906) -> dict[str, float]:
    return _floats(_finite_blob("hensel_lift", bench_hensel_lift(seed)))


def bench_poly_crt_family(seed: int = _SEED + 1907) -> dict[str, float]:
    return _floats(_finite_blob("poly_crt", bench_poly_crt(seed)))


def bench_subresultant_family(seed: int = _SEED + 1908) -> dict[str, float]:
    return _floats(_finite_blob("subresultant", bench_subresultant(seed)))


def bench_sparse_interp_family(seed: int = _SEED + 1909) -> dict[str, float]:
    return _floats(_finite_blob("sparse_interp", bench_sparse_interp(seed)))


def bench_poly_eval_interp_family(seed: int = _SEED + 1910) -> dict[str, float]:
    return _floats(_finite_blob("poly_eval_interp", bench_poly_eval_interp(seed)))
