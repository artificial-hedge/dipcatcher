"""Wave-1124 sociology-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.sociology_of_aging import bench_sociology_of_aging
from quant_fund.models.sociology_of_emotions import bench_sociology_of_emotions
from quant_fund.models.sociology_of_food import bench_sociology_of_food
from quant_fund.models.sociology_of_media import bench_sociology_of_media
from quant_fund.models.sociology_of_sport import bench_sociology_of_sport
from quant_fund.models.sociology_of_work import bench_sociology_of_work

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


def bench_sociology_of_work_family(seed: int = _SEED + 51200) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_work(seed))


def bench_sociology_of_emotions_family(seed: int = _SEED + 51201) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_emotions(seed))


def bench_sociology_of_food_family(seed: int = _SEED + 51202) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_food(seed))


def bench_sociology_of_media_family(seed: int = _SEED + 51203) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_media(seed))


def bench_sociology_of_sport_family(seed: int = _SEED + 51204) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_sport(seed))


def bench_sociology_of_aging_family(seed: int = _SEED + 51205) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_aging(seed))
