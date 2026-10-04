"""Wave-1126 psychology-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.community_psychology import bench_community_psychology
from quant_fund.models.consumer_psychology import bench_consumer_psychology
from quant_fund.models.cross_cultural_psychology import bench_cross_cultural_psychology
from quant_fund.models.political_psychology import bench_political_psychology
from quant_fund.models.positive_psychology import bench_positive_psychology
from quant_fund.models.social_cognition import bench_social_cognition

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


def bench_social_cognition_family(seed: int = _SEED + 51400) -> dict[str, float]:
    return _finite_blob(bench_social_cognition(seed))


def bench_positive_psychology_family(seed: int = _SEED + 51401) -> dict[str, float]:
    return _finite_blob(bench_positive_psychology(seed))


def bench_cross_cultural_psychology_family(seed: int = _SEED + 51402) -> dict[str, float]:
    return _finite_blob(bench_cross_cultural_psychology(seed))


def bench_consumer_psychology_family(seed: int = _SEED + 51403) -> dict[str, float]:
    return _finite_blob(bench_consumer_psychology(seed))


def bench_political_psychology_family(seed: int = _SEED + 51404) -> dict[str, float]:
    return _finite_blob(bench_political_psychology(seed))


def bench_community_psychology_family(seed: int = _SEED + 51405) -> dict[str, float]:
    return _finite_blob(bench_community_psychology(seed))
