"""Wave-946 positive-matrix canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cholesky_piv import bench_cholesky_piv
from quant_fund.models.douglas_factor import bench_douglas_factor
from quant_fund.models.matrix_square_root import bench_matrix_square_root
from quant_fund.models.perron_frobenius import bench_perron_frobenius
from quant_fund.models.polar_decomp import bench_polar_decomp
from quant_fund.models.sylvester_matrix import bench_sylvester_matrix

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


def bench_perron_frobenius_family(seed: int = _SEED + 33400) -> dict[str, float]:
    return _finite_blob(bench_perron_frobenius(seed))


def bench_douglas_factor_family(seed: int = _SEED + 33401) -> dict[str, float]:
    return _finite_blob(bench_douglas_factor(seed))


def bench_cholesky_piv_family(seed: int = _SEED + 33402) -> dict[str, float]:
    return _finite_blob(bench_cholesky_piv(seed))


def bench_matrix_square_root_family(seed: int = _SEED + 33403) -> dict[str, float]:
    return _finite_blob(bench_matrix_square_root(seed))


def bench_polar_decomp_family(seed: int = _SEED + 33404) -> dict[str, float]:
    return _finite_blob(bench_polar_decomp(seed))


def bench_sylvester_matrix_family(seed: int = _SEED + 33405) -> dict[str, float]:
    return _finite_blob(bench_sylvester_matrix(seed))
