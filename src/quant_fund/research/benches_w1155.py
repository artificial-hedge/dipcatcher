"""Wave-1155 earth-systems canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.atmospheric_science import bench_atmospheric_science
from quant_fund.models.earth_system_science import bench_earth_system_science
from quant_fund.models.environmental_science_2 import bench_environmental_science_2
from quant_fund.models.hydrology_3 import bench_hydrology_3
from quant_fund.models.oceanography_2 import bench_oceanography_2
from quant_fund.models.soil_science_2 import bench_soil_science_2

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


def bench_earth_system_science_family(seed: int = _SEED + 54300) -> dict[str, float]:
    return _finite_blob(bench_earth_system_science(seed))


def bench_oceanography_2_family(seed: int = _SEED + 54301) -> dict[str, float]:
    return _finite_blob(bench_oceanography_2(seed))


def bench_atmospheric_science_family(seed: int = _SEED + 54302) -> dict[str, float]:
    return _finite_blob(bench_atmospheric_science(seed))


def bench_environmental_science_2_family(seed: int = _SEED + 54303) -> dict[str, float]:
    return _finite_blob(bench_environmental_science_2(seed))


def bench_soil_science_2_family(seed: int = _SEED + 54304) -> dict[str, float]:
    return _finite_blob(bench_soil_science_2(seed))


def bench_hydrology_3_family(seed: int = _SEED + 54305) -> dict[str, float]:
    return _finite_blob(bench_hydrology_3(seed))
