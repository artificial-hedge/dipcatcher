"""Wave-1055 anthropology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.archaeology import bench_archaeology
from quant_fund.models.cultural_anthropology import bench_cultural_anthropology
from quant_fund.models.ethnography import bench_ethnography
from quant_fund.models.linguistic_anthropology import bench_linguistic_anthropology
from quant_fund.models.physical_anthropology import bench_physical_anthropology
from quant_fund.models.primatology import bench_primatology

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


def bench_physical_anthropology_family(seed: int = _SEED + 44300) -> dict[str, float]:
    return _finite_blob(bench_physical_anthropology(seed))


def bench_cultural_anthropology_family(seed: int = _SEED + 44301) -> dict[str, float]:
    return _finite_blob(bench_cultural_anthropology(seed))


def bench_archaeology_family(seed: int = _SEED + 44302) -> dict[str, float]:
    return _finite_blob(bench_archaeology(seed))


def bench_linguistic_anthropology_family(seed: int = _SEED + 44303) -> dict[str, float]:
    return _finite_blob(bench_linguistic_anthropology(seed))


def bench_primatology_family(seed: int = _SEED + 44304) -> dict[str, float]:
    return _finite_blob(bench_primatology(seed))


def bench_ethnography_family(seed: int = _SEED + 44305) -> dict[str, float]:
    return _finite_blob(bench_ethnography(seed))
