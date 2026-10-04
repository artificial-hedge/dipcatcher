"""Wave-1074 culinary arts canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.baking_science import bench_baking_science
from quant_fund.models.culinary_arts import bench_culinary_arts
from quant_fund.models.fermentation_science import bench_fermentation_science
from quant_fund.models.flavor_science import bench_flavor_science
from quant_fund.models.food_studies import bench_food_studies
from quant_fund.models.gastronomy import bench_gastronomy

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


def bench_culinary_arts_family(seed: int = _SEED + 46200) -> dict[str, float]:
    return _finite_blob(bench_culinary_arts(seed))


def bench_gastronomy_family(seed: int = _SEED + 46201) -> dict[str, float]:
    return _finite_blob(bench_gastronomy(seed))


def bench_food_studies_family(seed: int = _SEED + 46202) -> dict[str, float]:
    return _finite_blob(bench_food_studies(seed))


def bench_baking_science_family(seed: int = _SEED + 46203) -> dict[str, float]:
    return _finite_blob(bench_baking_science(seed))


def bench_flavor_science_family(seed: int = _SEED + 46204) -> dict[str, float]:
    return _finite_blob(bench_flavor_science(seed))


def bench_fermentation_science_family(seed: int = _SEED + 46205) -> dict[str, float]:
    return _finite_blob(bench_fermentation_science(seed))
