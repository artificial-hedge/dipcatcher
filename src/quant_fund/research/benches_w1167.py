"""Wave-1167 performing-arts canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.art_history_2 import bench_art_history_2
from quant_fund.models.dance_2 import bench_dance_2
from quant_fund.models.film_studies_3 import bench_film_studies_3
from quant_fund.models.music_2 import bench_music_2
from quant_fund.models.performance_studies_2 import bench_performance_studies_2
from quant_fund.models.theater_2 import bench_theater_2

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


def bench_music_2_family(seed: int = _SEED + 55500) -> dict[str, float]:
    return _finite_blob(bench_music_2(seed))


def bench_theater_2_family(seed: int = _SEED + 55501) -> dict[str, float]:
    return _finite_blob(bench_theater_2(seed))


def bench_dance_2_family(seed: int = _SEED + 55502) -> dict[str, float]:
    return _finite_blob(bench_dance_2(seed))


def bench_film_studies_3_family(seed: int = _SEED + 55503) -> dict[str, float]:
    return _finite_blob(bench_film_studies_3(seed))


def bench_art_history_2_family(seed: int = _SEED + 55504) -> dict[str, float]:
    return _finite_blob(bench_art_history_2(seed))


def bench_performance_studies_2_family(seed: int = _SEED + 55505) -> dict[str, float]:
    return _finite_blob(bench_performance_studies_2(seed))
