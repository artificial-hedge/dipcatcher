"""Wave-935 convex-optimization canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.analytic_center import bench_analytic_center
from quant_fund.models.cvx_reform import bench_cvx_reform
from quant_fund.models.dik_ellipsoid import bench_dik_ellipsoid
from quant_fund.models.kkt_solve import bench_kkt_solve
from quant_fund.models.logbarrier_fn import bench_logbarrier_fn
from quant_fund.models.self_concordant import bench_self_concordant

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


def bench_kkt_solve_family(seed: int = _SEED + 32300) -> dict[str, float]:
    return _finite_blob(bench_kkt_solve(seed))


def bench_cvx_reform_family(seed: int = _SEED + 32301) -> dict[str, float]:
    return _finite_blob(bench_cvx_reform(seed))


def bench_self_concordant_family(seed: int = _SEED + 32302) -> dict[str, float]:
    return _finite_blob(bench_self_concordant(seed))


def bench_logbarrier_fn_family(seed: int = _SEED + 32303) -> dict[str, float]:
    return _finite_blob(bench_logbarrier_fn(seed))


def bench_analytic_center_family(seed: int = _SEED + 32304) -> dict[str, float]:
    return _finite_blob(bench_analytic_center(seed))


def bench_dik_ellipsoid_family(seed: int = _SEED + 32305) -> dict[str, float]:
    return _finite_blob(bench_dik_ellipsoid(seed))
