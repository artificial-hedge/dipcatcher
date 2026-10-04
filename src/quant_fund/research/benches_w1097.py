"""Wave-1097 philosophy-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.eastern_philosophy import bench_eastern_philosophy
from quant_fund.models.moral_philosophy import bench_moral_philosophy
from quant_fund.models.philosophy_of_language import bench_philosophy_of_language
from quant_fund.models.philosophy_of_law import bench_philosophy_of_law
from quant_fund.models.philosophy_of_mind import bench_philosophy_of_mind
from quant_fund.models.political_philosophy import bench_political_philosophy

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


def bench_moral_philosophy_family(seed: int = _SEED + 48500) -> dict[str, float]:
    return _finite_blob(bench_moral_philosophy(seed))


def bench_political_philosophy_family(seed: int = _SEED + 48501) -> dict[str, float]:
    return _finite_blob(bench_political_philosophy(seed))


def bench_philosophy_of_mind_family(seed: int = _SEED + 48502) -> dict[str, float]:
    return _finite_blob(bench_philosophy_of_mind(seed))


def bench_philosophy_of_language_family(seed: int = _SEED + 48503) -> dict[str, float]:
    return _finite_blob(bench_philosophy_of_language(seed))


def bench_philosophy_of_law_family(seed: int = _SEED + 48504) -> dict[str, float]:
    return _finite_blob(bench_philosophy_of_law(seed))


def bench_eastern_philosophy_family(seed: int = _SEED + 48505) -> dict[str, float]:
    return _finite_blob(bench_eastern_philosophy(seed))
