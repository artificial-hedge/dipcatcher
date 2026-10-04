"""Wave-1108 sociology-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cultural_sociology import bench_cultural_sociology
from quant_fund.models.environmental_sociology import bench_environmental_sociology
from quant_fund.models.industrial_sociology import bench_industrial_sociology
from quant_fund.models.political_sociology import bench_political_sociology
from quant_fund.models.sociology_of_education import bench_sociology_of_education
from quant_fund.models.sociology_of_religion import bench_sociology_of_religion

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


def bench_industrial_sociology_family(seed: int = _SEED + 49600) -> dict[str, float]:
    return _finite_blob(bench_industrial_sociology(seed))


def bench_political_sociology_family(seed: int = _SEED + 49601) -> dict[str, float]:
    return _finite_blob(bench_political_sociology(seed))


def bench_sociology_of_education_family(seed: int = _SEED + 49602) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_education(seed))


def bench_sociology_of_religion_family(seed: int = _SEED + 49603) -> dict[str, float]:
    return _finite_blob(bench_sociology_of_religion(seed))


def bench_environmental_sociology_family(seed: int = _SEED + 49604) -> dict[str, float]:
    return _finite_blob(bench_environmental_sociology(seed))


def bench_cultural_sociology_family(seed: int = _SEED + 49605) -> dict[str, float]:
    return _finite_blob(bench_cultural_sociology(seed))
