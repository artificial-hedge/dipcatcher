"""Wave-1119 history-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cultural_history import bench_cultural_history
from quant_fund.models.diplomatic_history import bench_diplomatic_history
from quant_fund.models.history_of_medicine import bench_history_of_medicine
from quant_fund.models.history_of_technology import bench_history_of_technology
from quant_fund.models.military_history import bench_military_history
from quant_fund.models.social_history import bench_social_history

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


def bench_social_history_family(seed: int = _SEED + 50700) -> dict[str, float]:
    return _finite_blob(bench_social_history(seed))


def bench_cultural_history_family(seed: int = _SEED + 50701) -> dict[str, float]:
    return _finite_blob(bench_cultural_history(seed))


def bench_military_history_family(seed: int = _SEED + 50702) -> dict[str, float]:
    return _finite_blob(bench_military_history(seed))


def bench_diplomatic_history_family(seed: int = _SEED + 50703) -> dict[str, float]:
    return _finite_blob(bench_diplomatic_history(seed))


def bench_history_of_technology_family(seed: int = _SEED + 50704) -> dict[str, float]:
    return _finite_blob(bench_history_of_technology(seed))


def bench_history_of_medicine_family(seed: int = _SEED + 50705) -> dict[str, float]:
    return _finite_blob(bench_history_of_medicine(seed))
