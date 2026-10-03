"""Wave-923 Bayesian-nonparametrics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.chinese_restaurant import bench_chinese_restaurant
from quant_fund.models.dirichlet_process import bench_dirichlet_process
from quant_fund.models.hierarchical_dp import bench_hierarchical_dp
from quant_fund.models.indian_buffet import bench_indian_buffet
from quant_fund.models.pitman_yor import bench_pitman_yor
from quant_fund.models.stick_breaking import bench_stick_breaking

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


def bench_dirichlet_process_family(seed: int = _SEED + 31100) -> dict[str, float]:
    return _finite_blob(bench_dirichlet_process(seed))


def bench_stick_breaking_family(seed: int = _SEED + 31101) -> dict[str, float]:
    return _finite_blob(bench_stick_breaking(seed))


def bench_pitman_yor_family(seed: int = _SEED + 31102) -> dict[str, float]:
    return _finite_blob(bench_pitman_yor(seed))


def bench_indian_buffet_family(seed: int = _SEED + 31103) -> dict[str, float]:
    return _finite_blob(bench_indian_buffet(seed))


def bench_chinese_restaurant_family(seed: int = _SEED + 31104) -> dict[str, float]:
    return _finite_blob(bench_chinese_restaurant(seed))


def bench_hierarchical_dp_family(seed: int = _SEED + 31105) -> dict[str, float]:
    return _finite_blob(bench_hierarchical_dp(seed))
