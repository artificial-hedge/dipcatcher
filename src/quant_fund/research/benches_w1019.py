"""Wave-1019 geophysics-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.earthquake_magnitude import bench_earthquake_magnitude
from quant_fund.models.geomagnetism import bench_geomagnetism
from quant_fund.models.gravity_anomaly import bench_gravity_anomaly
from quant_fund.models.heat_flow_geo import bench_heat_flow_geo
from quant_fund.models.plate_tectonics import bench_plate_tectonics
from quant_fund.models.seismic_waves import bench_seismic_waves

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


def bench_seismic_waves_family(seed: int = _SEED + 40700) -> dict[str, float]:
    return _finite_blob(bench_seismic_waves(seed))


def bench_earthquake_magnitude_family(seed: int = _SEED + 40701) -> dict[str, float]:
    return _finite_blob(bench_earthquake_magnitude(seed))


def bench_plate_tectonics_family(seed: int = _SEED + 40702) -> dict[str, float]:
    return _finite_blob(bench_plate_tectonics(seed))


def bench_gravity_anomaly_family(seed: int = _SEED + 40703) -> dict[str, float]:
    return _finite_blob(bench_gravity_anomaly(seed))


def bench_geomagnetism_family(seed: int = _SEED + 40704) -> dict[str, float]:
    return _finite_blob(bench_geomagnetism(seed))


def bench_heat_flow_geo_family(seed: int = _SEED + 40705) -> dict[str, float]:
    return _finite_blob(bench_heat_flow_geo(seed))
