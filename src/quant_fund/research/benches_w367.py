"""Wave-367 Galois-2/field-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cyclotomic_poly import bench_cyclotomic_poly
from quant_fund.models.finite_field import bench_finite_field
from quant_fund.models.galois_corresp import bench_galois_corresp
from quant_fund.models.normality_check import bench_normality_check
from quant_fund.models.primitive_elem import bench_primitive_elem
from quant_fund.models.separable_check import bench_separable_check

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


def bench_finite_field_family(seed: int = _SEED + 2109) -> dict[str, float]:
    return _floats(_finite_blob("finite_field", bench_finite_field(seed)))


def bench_galois_corresp_family(seed: int = _SEED + 2110) -> dict[str, float]:
    return _floats(_finite_blob("galois_corresp", bench_galois_corresp(seed)))


def bench_normality_check_family(seed: int = _SEED + 2111) -> dict[str, float]:
    return _floats(_finite_blob("normality_check", bench_normality_check(seed)))


def bench_separable_check_family(seed: int = _SEED + 2112) -> dict[str, float]:
    return _floats(_finite_blob("separable_check", bench_separable_check(seed)))


def bench_cyclotomic_poly_family(seed: int = _SEED + 2113) -> dict[str, float]:
    return _floats(_finite_blob("cyclotomic_poly", bench_cyclotomic_poly(seed)))


def bench_primitive_elem_family(seed: int = _SEED + 2114) -> dict[str, float]:
    return _floats(_finite_blob("primitive_elem", bench_primitive_elem(seed)))
