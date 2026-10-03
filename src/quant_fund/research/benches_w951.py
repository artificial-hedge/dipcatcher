"""Wave-951 matrix-function canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.determinant_cofactor import bench_determinant_cofactor
from quant_fund.models.frechet_derivative import bench_frechet_derivative
from quant_fund.models.kronecker_sum import bench_kronecker_sum
from quant_fund.models.matrix_exponential import bench_matrix_exponential
from quant_fund.models.permanent_matrix import bench_permanent_matrix
from quant_fund.models.vec_operator import bench_vec_operator

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_determinant_cofactor_family(seed: int = _SEED + 33900) -> dict[str, float]:
    return _finite_blob(bench_determinant_cofactor(seed))


def bench_permanent_matrix_family(seed: int = _SEED + 33901) -> dict[str, float]:
    return _finite_blob(bench_permanent_matrix(seed))


def bench_matrix_exponential_family(seed: int = _SEED + 33902) -> dict[str, float]:
    return _finite_blob(bench_matrix_exponential(seed))


def bench_frechet_derivative_family(seed: int = _SEED + 33903) -> dict[str, float]:
    return _finite_blob(bench_frechet_derivative(seed))


def bench_vec_operator_family(seed: int = _SEED + 33904) -> dict[str, float]:
    return _finite_blob(bench_vec_operator(seed))


def bench_kronecker_sum_family(seed: int = _SEED + 33905) -> dict[str, float]:
    return _finite_blob(bench_kronecker_sum(seed))
