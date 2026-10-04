"""Wave-948 structured-matrix canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.circulant_matrix import bench_circulant_matrix
from quant_fund.models.companion_matrix import bench_companion_matrix
from quant_fund.models.hankel_matrix import bench_hankel_matrix
from quant_fund.models.hessenberg_form import bench_hessenberg_form
from quant_fund.models.krylov_matrix import bench_krylov_matrix
from quant_fund.models.vandermonde_matrix import bench_vandermonde_matrix

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


def bench_circulant_matrix_family(seed: int = _SEED + 33600) -> dict[str, float]:
    return _finite_blob(bench_circulant_matrix(seed))


def bench_companion_matrix_family(seed: int = _SEED + 33601) -> dict[str, float]:
    return _finite_blob(bench_companion_matrix(seed))


def bench_vandermonde_matrix_family(seed: int = _SEED + 33602) -> dict[str, float]:
    return _finite_blob(bench_vandermonde_matrix(seed))


def bench_krylov_matrix_family(seed: int = _SEED + 33603) -> dict[str, float]:
    return _finite_blob(bench_krylov_matrix(seed))


def bench_hessenberg_form_family(seed: int = _SEED + 33604) -> dict[str, float]:
    return _finite_blob(bench_hessenberg_form(seed))


def bench_hankel_matrix_family(seed: int = _SEED + 33605) -> dict[str, float]:
    return _finite_blob(bench_hankel_matrix(seed))
