"""Wave-1070 sports science canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.athletic_training import bench_athletic_training
from quant_fund.models.exercise_physiology import bench_exercise_physiology
from quant_fund.models.sports_analytics import bench_sports_analytics
from quant_fund.models.sports_biomechanics import bench_sports_biomechanics
from quant_fund.models.sports_psychology import bench_sports_psychology
from quant_fund.models.sports_science import bench_sports_science

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


def bench_sports_science_family(seed: int = _SEED + 45800) -> dict[str, float]:
    return _finite_blob(bench_sports_science(seed))


def bench_exercise_physiology_family(seed: int = _SEED + 45801) -> dict[str, float]:
    return _finite_blob(bench_exercise_physiology(seed))


def bench_sports_biomechanics_family(seed: int = _SEED + 45802) -> dict[str, float]:
    return _finite_blob(bench_sports_biomechanics(seed))


def bench_sports_psychology_family(seed: int = _SEED + 45803) -> dict[str, float]:
    return _finite_blob(bench_sports_psychology(seed))


def bench_athletic_training_family(seed: int = _SEED + 45804) -> dict[str, float]:
    return _finite_blob(bench_athletic_training(seed))


def bench_sports_analytics_family(seed: int = _SEED + 45805) -> dict[str, float]:
    return _finite_blob(bench_sports_analytics(seed))
