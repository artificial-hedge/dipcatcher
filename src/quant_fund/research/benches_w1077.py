"""Wave-1077 visual arts canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.art_conservation import bench_art_conservation
from quant_fund.models.art_history import bench_art_history
from quant_fund.models.painting_techniques import bench_painting_techniques
from quant_fund.models.printmaking import bench_printmaking
from quant_fund.models.sculpture_methods import bench_sculpture_methods
from quant_fund.models.visual_culture import bench_visual_culture

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


def bench_painting_techniques_family(seed: int = _SEED + 46500) -> dict[str, float]:
    return _finite_blob(bench_painting_techniques(seed))


def bench_sculpture_methods_family(seed: int = _SEED + 46501) -> dict[str, float]:
    return _finite_blob(bench_sculpture_methods(seed))


def bench_printmaking_family(seed: int = _SEED + 46502) -> dict[str, float]:
    return _finite_blob(bench_printmaking(seed))


def bench_art_conservation_family(seed: int = _SEED + 46503) -> dict[str, float]:
    return _finite_blob(bench_art_conservation(seed))


def bench_art_history_family(seed: int = _SEED + 46504) -> dict[str, float]:
    return _finite_blob(bench_art_history(seed))


def bench_visual_culture_family(seed: int = _SEED + 46505) -> dict[str, float]:
    return _finite_blob(bench_visual_culture(seed))
