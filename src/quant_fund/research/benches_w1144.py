"""Wave-1144 space-science canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.asteroid_science import bench_asteroid_science
from quant_fund.models.astrophotonics import bench_astrophotonics
from quant_fund.models.comet_science import bench_comet_science
from quant_fund.models.grav_waves_2 import bench_grav_waves_2
from quant_fund.models.planetology import bench_planetology
from quant_fund.models.space_weather import bench_space_weather

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


def bench_space_weather_family(seed: int = _SEED + 53200) -> dict[str, float]:
    return _finite_blob(bench_space_weather(seed))


def bench_planetology_family(seed: int = _SEED + 53201) -> dict[str, float]:
    return _finite_blob(bench_planetology(seed))


def bench_asteroid_science_family(seed: int = _SEED + 53202) -> dict[str, float]:
    return _finite_blob(bench_asteroid_science(seed))


def bench_comet_science_family(seed: int = _SEED + 53203) -> dict[str, float]:
    return _finite_blob(bench_comet_science(seed))


def bench_astrophotonics_family(seed: int = _SEED + 53204) -> dict[str, float]:
    return _finite_blob(bench_astrophotonics(seed))


def bench_grav_waves_2_family(seed: int = _SEED + 53205) -> dict[str, float]:
    return _finite_blob(bench_grav_waves_2(seed))
