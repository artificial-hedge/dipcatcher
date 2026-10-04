"""Wave-1020 ecology/evolution canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.food_web import bench_food_web
from quant_fund.models.island_biogeography import bench_island_biogeography
from quant_fund.models.logistic_growth import bench_logistic_growth
from quant_fund.models.lotka_volterra import bench_lotka_volterra
from quant_fund.models.neutral_theory import bench_neutral_theory
from quant_fund.models.predator_prey import bench_predator_prey

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


def bench_predator_prey_family(seed: int = _SEED + 40800) -> dict[str, float]:
    return _finite_blob(bench_predator_prey(seed))


def bench_lotka_volterra_family(seed: int = _SEED + 40801) -> dict[str, float]:
    return _finite_blob(bench_lotka_volterra(seed))


def bench_logistic_growth_family(seed: int = _SEED + 40802) -> dict[str, float]:
    return _finite_blob(bench_logistic_growth(seed))


def bench_island_biogeography_family(seed: int = _SEED + 40803) -> dict[str, float]:
    return _finite_blob(bench_island_biogeography(seed))


def bench_neutral_theory_family(seed: int = _SEED + 40804) -> dict[str, float]:
    return _finite_blob(bench_neutral_theory(seed))


def bench_food_web_family(seed: int = _SEED + 40805) -> dict[str, float]:
    return _finite_blob(bench_food_web(seed))
