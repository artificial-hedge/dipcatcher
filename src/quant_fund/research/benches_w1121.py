"""Wave-1121 geography-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.economic_geography import bench_economic_geography
from quant_fund.models.gis_science import bench_gis_science
from quant_fund.models.health_geography import bench_health_geography
from quant_fund.models.political_geography import bench_political_geography
from quant_fund.models.population_geography import bench_population_geography
from quant_fund.models.regional_geography import bench_regional_geography

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


def bench_regional_geography_family(seed: int = _SEED + 50900) -> dict[str, float]:
    return _finite_blob(bench_regional_geography(seed))


def bench_health_geography_family(seed: int = _SEED + 50901) -> dict[str, float]:
    return _finite_blob(bench_health_geography(seed))


def bench_population_geography_family(seed: int = _SEED + 50902) -> dict[str, float]:
    return _finite_blob(bench_population_geography(seed))


def bench_economic_geography_family(seed: int = _SEED + 50903) -> dict[str, float]:
    return _finite_blob(bench_economic_geography(seed))


def bench_political_geography_family(seed: int = _SEED + 50904) -> dict[str, float]:
    return _finite_blob(bench_political_geography(seed))


def bench_gis_science_family(seed: int = _SEED + 50905) -> dict[str, float]:
    return _finite_blob(bench_gis_science(seed))
