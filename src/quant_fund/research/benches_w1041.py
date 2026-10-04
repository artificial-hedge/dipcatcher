"""Wave-1041 ocean-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.coastal_engineering import bench_coastal_engineering
from quant_fund.models.marine_propulsion import bench_marine_propulsion
from quant_fund.models.naval_architecture import bench_naval_architecture
from quant_fund.models.ocean_waves import bench_ocean_waves
from quant_fund.models.offshore_engineering import bench_offshore_engineering
from quant_fund.models.submarine_systems import bench_submarine_systems

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


def bench_naval_architecture_family(seed: int = _SEED + 42900) -> dict[str, float]:
    return _finite_blob(bench_naval_architecture(seed))


def bench_offshore_engineering_family(seed: int = _SEED + 42901) -> dict[str, float]:
    return _finite_blob(bench_offshore_engineering(seed))


def bench_marine_propulsion_family(seed: int = _SEED + 42902) -> dict[str, float]:
    return _finite_blob(bench_marine_propulsion(seed))


def bench_ocean_waves_family(seed: int = _SEED + 42903) -> dict[str, float]:
    return _finite_blob(bench_ocean_waves(seed))


def bench_coastal_engineering_family(seed: int = _SEED + 42904) -> dict[str, float]:
    return _finite_blob(bench_coastal_engineering(seed))


def bench_submarine_systems_family(seed: int = _SEED + 42905) -> dict[str, float]:
    return _finite_blob(bench_submarine_systems(seed))
