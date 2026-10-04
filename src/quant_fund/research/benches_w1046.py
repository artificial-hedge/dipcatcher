"""Wave-1046 meteorology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.atmospheric_dynamics import bench_atmospheric_dynamics
from quant_fund.models.climate_dynamics import bench_climate_dynamics
from quant_fund.models.cloud_physics import bench_cloud_physics
from quant_fund.models.mesoscale_meteorology import bench_mesoscale_meteorology
from quant_fund.models.numerical_weather import bench_numerical_weather
from quant_fund.models.synoptic_meteorology import bench_synoptic_meteorology

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


def bench_atmospheric_dynamics_family(seed: int = _SEED + 43400) -> dict[str, float]:
    return _finite_blob(bench_atmospheric_dynamics(seed))


def bench_synoptic_meteorology_family(seed: int = _SEED + 43401) -> dict[str, float]:
    return _finite_blob(bench_synoptic_meteorology(seed))


def bench_cloud_physics_family(seed: int = _SEED + 43402) -> dict[str, float]:
    return _finite_blob(bench_cloud_physics(seed))


def bench_numerical_weather_family(seed: int = _SEED + 43403) -> dict[str, float]:
    return _finite_blob(bench_numerical_weather(seed))


def bench_mesoscale_meteorology_family(seed: int = _SEED + 43404) -> dict[str, float]:
    return _finite_blob(bench_mesoscale_meteorology(seed))


def bench_climate_dynamics_family(seed: int = _SEED + 43405) -> dict[str, float]:
    return _finite_blob(bench_climate_dynamics(seed))
