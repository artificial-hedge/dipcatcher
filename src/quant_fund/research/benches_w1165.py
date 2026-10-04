"""Wave-1165 agriculture canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.agriculture_2 import bench_agriculture_2
from quant_fund.models.fisheries_2 import bench_fisheries_2
from quant_fund.models.food_science_2 import bench_food_science_2
from quant_fund.models.forestry_2 import bench_forestry_2
from quant_fund.models.horticulture_2 import bench_horticulture_2
from quant_fund.models.veterinary_science_2 import bench_veterinary_science_2

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


def bench_agriculture_2_family(seed: int = _SEED + 55300) -> dict[str, float]:
    return _finite_blob(bench_agriculture_2(seed))


def bench_food_science_2_family(seed: int = _SEED + 55301) -> dict[str, float]:
    return _finite_blob(bench_food_science_2(seed))


def bench_forestry_2_family(seed: int = _SEED + 55302) -> dict[str, float]:
    return _finite_blob(bench_forestry_2(seed))


def bench_fisheries_2_family(seed: int = _SEED + 55303) -> dict[str, float]:
    return _finite_blob(bench_fisheries_2(seed))


def bench_horticulture_2_family(seed: int = _SEED + 55304) -> dict[str, float]:
    return _finite_blob(bench_horticulture_2(seed))


def bench_veterinary_science_2_family(seed: int = _SEED + 55305) -> dict[str, float]:
    return _finite_blob(bench_veterinary_science_2(seed))
