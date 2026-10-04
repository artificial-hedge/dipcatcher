"""Wave-1160 humanities canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.area_studies_2 import bench_area_studies_2
from quant_fund.models.classics_2 import bench_classics_2
from quant_fund.models.history_5 import bench_history_5
from quant_fund.models.humanities_2 import bench_humanities_2
from quant_fund.models.philosophy_6 import bench_philosophy_6
from quant_fund.models.religious_studies_2 import bench_religious_studies_2

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


def bench_philosophy_6_family(seed: int = _SEED + 54800) -> dict[str, float]:
    return _finite_blob(bench_philosophy_6(seed))


def bench_history_5_family(seed: int = _SEED + 54801) -> dict[str, float]:
    return _finite_blob(bench_history_5(seed))


def bench_religious_studies_2_family(seed: int = _SEED + 54802) -> dict[str, float]:
    return _finite_blob(bench_religious_studies_2(seed))


def bench_classics_2_family(seed: int = _SEED + 54803) -> dict[str, float]:
    return _finite_blob(bench_classics_2(seed))


def bench_area_studies_2_family(seed: int = _SEED + 54804) -> dict[str, float]:
    return _finite_blob(bench_area_studies_2(seed))


def bench_humanities_2_family(seed: int = _SEED + 54805) -> dict[str, float]:
    return _finite_blob(bench_humanities_2(seed))
