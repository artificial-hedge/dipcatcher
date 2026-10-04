"""Wave-990 calculus-of-variations canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.euler_lagrange import bench_euler_lagrange
from quant_fund.models.geodesic_var import bench_geodesic_var
from quant_fund.models.isoperimetric_var import bench_isoperimetric_var
from quant_fund.models.jacobi_eq import bench_jacobi_eq
from quant_fund.models.legendre_cond import bench_legendre_cond
from quant_fund.models.soap_film import bench_soap_film

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


def bench_euler_lagrange_family(seed: int = _SEED + 37800) -> dict[str, float]:
    return _finite_blob(bench_euler_lagrange(seed))


def bench_legendre_cond_family(seed: int = _SEED + 37801) -> dict[str, float]:
    return _finite_blob(bench_legendre_cond(seed))


def bench_jacobi_eq_family(seed: int = _SEED + 37802) -> dict[str, float]:
    return _finite_blob(bench_jacobi_eq(seed))


def bench_geodesic_var_family(seed: int = _SEED + 37803) -> dict[str, float]:
    return _finite_blob(bench_geodesic_var(seed))


def bench_isoperimetric_var_family(seed: int = _SEED + 37804) -> dict[str, float]:
    return _finite_blob(bench_isoperimetric_var(seed))


def bench_soap_film_family(seed: int = _SEED + 37805) -> dict[str, float]:
    return _finite_blob(bench_soap_film(seed))
