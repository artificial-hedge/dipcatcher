"""Wave-430 higher-algebra bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.brane_tensor import bench_brane_tensor
from quant_fund.models.delooping import bench_delooping
from quant_fund.models.e_n_algebra import bench_e_n_algebra
from quant_fund.models.module_cat import bench_module_cat
from quant_fund.models.monoidal_infty import bench_monoidal_infty
from quant_fund.models.operad_infty import bench_operad_infty

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


def bench_e_n_algebra_family(
    seed: int = _SEED + 2486,
) -> dict[str, float]:
    return _floats(_finite_blob("e_n_algebra", bench_e_n_algebra(seed)))


def bench_operad_infty_family(
    seed: int = _SEED + 2487,
) -> dict[str, float]:
    return _floats(_finite_blob("operad_infty", bench_operad_infty(seed)))


def bench_monoidal_infty_family(
    seed: int = _SEED + 2488,
) -> dict[str, float]:
    return _floats(_finite_blob("monoidal_infty", bench_monoidal_infty(seed)))


def bench_module_cat_family(
    seed: int = _SEED + 2489,
) -> dict[str, float]:
    return _floats(_finite_blob("module_cat", bench_module_cat(seed)))


def bench_brane_tensor_family(
    seed: int = _SEED + 2490,
) -> dict[str, float]:
    return _floats(_finite_blob("brane_tensor", bench_brane_tensor(seed)))


def bench_delooping_family(
    seed: int = _SEED + 2491,
) -> dict[str, float]:
    return _floats(_finite_blob("delooping", bench_delooping(seed)))
