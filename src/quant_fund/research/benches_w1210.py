"""Wave-1210 psychiatry canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.anxiety_disorders import bench_anxiety_disorders
from quant_fund.models.forensic_psychiatry import bench_forensic_psychiatry
from quant_fund.models.geriatric_psychiatry import bench_geriatric_psychiatry
from quant_fund.models.mood_disorders import bench_mood_disorders
from quant_fund.models.personality_disorders import bench_personality_disorders
from quant_fund.models.psychotic_disorders import bench_psychotic_disorders

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


def bench_forensic_psychiatry_family(seed: int = _SEED + 59800) -> dict[str, float]:
    return _finite_blob(bench_forensic_psychiatry(seed))


def bench_geriatric_psychiatry_family(seed: int = _SEED + 59801) -> dict[str, float]:
    return _finite_blob(bench_geriatric_psychiatry(seed))


def bench_mood_disorders_family(seed: int = _SEED + 59802) -> dict[str, float]:
    return _finite_blob(bench_mood_disorders(seed))


def bench_psychotic_disorders_family(seed: int = _SEED + 59803) -> dict[str, float]:
    return _finite_blob(bench_psychotic_disorders(seed))


def bench_personality_disorders_family(seed: int = _SEED + 59804) -> dict[str, float]:
    return _finite_blob(bench_personality_disorders(seed))


def bench_anxiety_disorders_family(seed: int = _SEED + 59805) -> dict[str, float]:
    return _finite_blob(bench_anxiety_disorders(seed))
