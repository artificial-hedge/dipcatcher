"""Wave-1103 meteorology-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.boundary_layer_meteorology import bench_boundary_layer_meteorology
from quant_fund.models.micrometeorology import bench_micrometeorology
from quant_fund.models.polar_meteorology import bench_polar_meteorology
from quant_fund.models.radar_meteorology import bench_radar_meteorology
from quant_fund.models.severe_weather import bench_severe_weather
from quant_fund.models.tropical_meteorology import bench_tropical_meteorology

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


def bench_severe_weather_family(seed: int = _SEED + 49100) -> dict[str, float]:
    return _finite_blob(bench_severe_weather(seed))


def bench_boundary_layer_meteorology_family(seed: int = _SEED + 49101) -> dict[str, float]:
    return _finite_blob(bench_boundary_layer_meteorology(seed))


def bench_radar_meteorology_family(seed: int = _SEED + 49102) -> dict[str, float]:
    return _finite_blob(bench_radar_meteorology(seed))


def bench_tropical_meteorology_family(seed: int = _SEED + 49103) -> dict[str, float]:
    return _finite_blob(bench_tropical_meteorology(seed))


def bench_polar_meteorology_family(seed: int = _SEED + 49104) -> dict[str, float]:
    return _finite_blob(bench_polar_meteorology(seed))


def bench_micrometeorology_family(seed: int = _SEED + 49105) -> dict[str, float]:
    return _finite_blob(bench_micrometeorology(seed))
