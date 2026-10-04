"""Wave-1171 geography canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.demography_2 import bench_demography_2
from quant_fund.models.geography_2 import bench_geography_2
from quant_fund.models.gis_science_2 import bench_gis_science_2
from quant_fund.models.land_use import bench_land_use
from quant_fund.models.regional_science import bench_regional_science
from quant_fund.models.urbanization import bench_urbanization

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


def bench_geography_2_family(seed: int = _SEED + 55900) -> dict[str, float]:
    return _finite_blob(bench_geography_2(seed))


def bench_regional_science_family(seed: int = _SEED + 55901) -> dict[str, float]:
    return _finite_blob(bench_regional_science(seed))


def bench_demography_2_family(seed: int = _SEED + 55902) -> dict[str, float]:
    return _finite_blob(bench_demography_2(seed))


def bench_urbanization_family(seed: int = _SEED + 55903) -> dict[str, float]:
    return _finite_blob(bench_urbanization(seed))


def bench_land_use_family(seed: int = _SEED + 55904) -> dict[str, float]:
    return _finite_blob(bench_land_use(seed))


def bench_gis_science_2_family(seed: int = _SEED + 55905) -> dict[str, float]:
    return _finite_blob(bench_gis_science_2(seed))
