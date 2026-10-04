"""Wave-1098 anthropology-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.biological_anthropology import bench_biological_anthropology
from quant_fund.models.economic_anthropology import bench_economic_anthropology
from quant_fund.models.medical_anthropology import bench_medical_anthropology
from quant_fund.models.paleoanthropology import bench_paleoanthropology
from quant_fund.models.political_anthropology import bench_political_anthropology
from quant_fund.models.urban_anthropology import bench_urban_anthropology

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


def bench_biological_anthropology_family(seed: int = _SEED + 48600) -> dict[str, float]:
    return _finite_blob(bench_biological_anthropology(seed))


def bench_paleoanthropology_family(seed: int = _SEED + 48601) -> dict[str, float]:
    return _finite_blob(bench_paleoanthropology(seed))


def bench_medical_anthropology_family(seed: int = _SEED + 48602) -> dict[str, float]:
    return _finite_blob(bench_medical_anthropology(seed))


def bench_economic_anthropology_family(seed: int = _SEED + 48603) -> dict[str, float]:
    return _finite_blob(bench_economic_anthropology(seed))


def bench_political_anthropology_family(seed: int = _SEED + 48604) -> dict[str, float]:
    return _finite_blob(bench_political_anthropology(seed))


def bench_urban_anthropology_family(seed: int = _SEED + 48605) -> dict[str, float]:
    return _finite_blob(bench_urban_anthropology(seed))
