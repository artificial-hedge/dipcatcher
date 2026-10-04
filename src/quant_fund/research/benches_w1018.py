"""Wave-1018 relativity-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.four_vectors import bench_four_vectors
from quant_fund.models.geodesic_motion import bench_geodesic_motion
from quant_fund.models.gravitational_lensing import bench_gravitational_lensing
from quant_fund.models.gravitational_waves import bench_gravitational_waves
from quant_fund.models.lorentz_transformation import bench_lorentz_transformation
from quant_fund.models.spacetime_interval import bench_spacetime_interval

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


def bench_lorentz_transformation_family(seed: int = _SEED + 40600) -> dict[str, float]:
    return _finite_blob(bench_lorentz_transformation(seed))


def bench_spacetime_interval_family(seed: int = _SEED + 40601) -> dict[str, float]:
    return _finite_blob(bench_spacetime_interval(seed))


def bench_four_vectors_family(seed: int = _SEED + 40602) -> dict[str, float]:
    return _finite_blob(bench_four_vectors(seed))


def bench_geodesic_motion_family(seed: int = _SEED + 40603) -> dict[str, float]:
    return _finite_blob(bench_geodesic_motion(seed))


def bench_gravitational_lensing_family(seed: int = _SEED + 40604) -> dict[str, float]:
    return _finite_blob(bench_gravitational_lensing(seed))


def bench_gravitational_waves_family(seed: int = _SEED + 40605) -> dict[str, float]:
    return _finite_blob(bench_gravitational_waves(seed))
