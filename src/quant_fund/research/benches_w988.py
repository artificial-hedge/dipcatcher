"""Wave-988 spectral-geometry canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cheeger_ineq import bench_cheeger_ineq
from quant_fund.models.heat_invariants import bench_heat_invariants
from quant_fund.models.isospectral import bench_isospectral
from quant_fund.models.nodal_domain import bench_nodal_domain
from quant_fund.models.spectral_geometry import bench_spectral_geometry
from quant_fund.models.weyl_law import bench_weyl_law

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


def bench_spectral_geometry_family(seed: int = _SEED + 37600) -> dict[str, float]:
    return _finite_blob(bench_spectral_geometry(seed))


def bench_heat_invariants_family(seed: int = _SEED + 37601) -> dict[str, float]:
    return _finite_blob(bench_heat_invariants(seed))


def bench_weyl_law_family(seed: int = _SEED + 37602) -> dict[str, float]:
    return _finite_blob(bench_weyl_law(seed))


def bench_isospectral_family(seed: int = _SEED + 37603) -> dict[str, float]:
    return _finite_blob(bench_isospectral(seed))


def bench_cheeger_ineq_family(seed: int = _SEED + 37604) -> dict[str, float]:
    return _finite_blob(bench_cheeger_ineq(seed))


def bench_nodal_domain_family(seed: int = _SEED + 37605) -> dict[str, float]:
    return _finite_blob(bench_nodal_domain(seed))
