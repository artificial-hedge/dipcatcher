"""Wave-1042 food-science canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.food_chemistry import bench_food_chemistry
from quant_fund.models.food_microbiology import bench_food_microbiology
from quant_fund.models.food_processing import bench_food_processing
from quant_fund.models.food_safety import bench_food_safety
from quant_fund.models.nutrition_science import bench_nutrition_science
from quant_fund.models.sensory_evaluation import bench_sensory_evaluation

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


def bench_food_chemistry_family(seed: int = _SEED + 43000) -> dict[str, float]:
    return _finite_blob(bench_food_chemistry(seed))


def bench_food_microbiology_family(seed: int = _SEED + 43001) -> dict[str, float]:
    return _finite_blob(bench_food_microbiology(seed))


def bench_food_processing_family(seed: int = _SEED + 43002) -> dict[str, float]:
    return _finite_blob(bench_food_processing(seed))


def bench_nutrition_science_family(seed: int = _SEED + 43003) -> dict[str, float]:
    return _finite_blob(bench_nutrition_science(seed))


def bench_sensory_evaluation_family(seed: int = _SEED + 43004) -> dict[str, float]:
    return _finite_blob(bench_sensory_evaluation(seed))


def bench_food_safety_family(seed: int = _SEED + 43005) -> dict[str, float]:
    return _finite_blob(bench_food_safety(seed))
