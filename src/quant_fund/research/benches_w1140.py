"""Wave-1140 earth-science canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.glaciology import bench_glaciology
from quant_fund.models.hydrology_2 import bench_hydrology_2
from quant_fund.models.oceanography import bench_oceanography
from quant_fund.models.paleoclimatology import bench_paleoclimatology
from quant_fund.models.seismology import bench_seismology
from quant_fund.models.volcanology_2 import bench_volcanology_2

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


def bench_oceanography_family(seed: int = _SEED + 52800) -> dict[str, float]:
    return _finite_blob(bench_oceanography(seed))


def bench_hydrology_2_family(seed: int = _SEED + 52801) -> dict[str, float]:
    return _finite_blob(bench_hydrology_2(seed))


def bench_seismology_family(seed: int = _SEED + 52802) -> dict[str, float]:
    return _finite_blob(bench_seismology(seed))


def bench_glaciology_family(seed: int = _SEED + 52803) -> dict[str, float]:
    return _finite_blob(bench_glaciology(seed))


def bench_paleoclimatology_family(seed: int = _SEED + 52804) -> dict[str, float]:
    return _finite_blob(bench_paleoclimatology(seed))


def bench_volcanology_2_family(seed: int = _SEED + 52805) -> dict[str, float]:
    return _finite_blob(bench_volcanology_2(seed))
