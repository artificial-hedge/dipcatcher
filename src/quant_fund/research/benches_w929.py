"""Wave-929 information-geometry-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bhat_distance import bench_bhat_distance
from quant_fund.models.chi_square_div import bench_chi_square_div
from quant_fund.models.d_total_var import bench_d_total_var
from quant_fund.models.hellinger_dist import bench_hellinger_dist
from quant_fund.models.jeffreys_div import bench_jeffreys_div
from quant_fund.models.mahalanobis_div import bench_mahalanobis_div

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


def bench_mahalanobis_div_family(seed: int = _SEED + 31700) -> dict[str, float]:
    return _finite_blob(bench_mahalanobis_div(seed))


def bench_bhat_distance_family(seed: int = _SEED + 31701) -> dict[str, float]:
    return _finite_blob(bench_bhat_distance(seed))


def bench_hellinger_dist_family(seed: int = _SEED + 31702) -> dict[str, float]:
    return _finite_blob(bench_hellinger_dist(seed))


def bench_jeffreys_div_family(seed: int = _SEED + 31703) -> dict[str, float]:
    return _finite_blob(bench_jeffreys_div(seed))


def bench_d_total_var_family(seed: int = _SEED + 31704) -> dict[str, float]:
    return _finite_blob(bench_d_total_var(seed))


def bench_chi_square_div_family(seed: int = _SEED + 31705) -> dict[str, float]:
    return _finite_blob(bench_chi_square_div(seed))
