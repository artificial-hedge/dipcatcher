"""Wave-1003 MHD/plasma canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.alfven_waves import bench_alfven_waves
from quant_fund.models.elsaesser_vars import bench_elsaesser_vars
from quant_fund.models.frozen_flux import bench_frozen_flux
from quant_fund.models.magnetic_reconnection import bench_magnetic_reconnection
from quant_fund.models.mhd_equations import bench_mhd_equations
from quant_fund.models.parker_solar_wind import bench_parker_solar_wind

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


def bench_mhd_equations_family(seed: int = _SEED + 39100) -> dict[str, float]:
    return _finite_blob(bench_mhd_equations(seed))


def bench_alfven_waves_family(seed: int = _SEED + 39101) -> dict[str, float]:
    return _finite_blob(bench_alfven_waves(seed))


def bench_parker_solar_wind_family(seed: int = _SEED + 39102) -> dict[str, float]:
    return _finite_blob(bench_parker_solar_wind(seed))


def bench_magnetic_reconnection_family(seed: int = _SEED + 39103) -> dict[str, float]:
    return _finite_blob(bench_magnetic_reconnection(seed))


def bench_frozen_flux_family(seed: int = _SEED + 39104) -> dict[str, float]:
    return _finite_blob(bench_frozen_flux(seed))


def bench_elsaesser_vars_family(seed: int = _SEED + 39105) -> dict[str, float]:
    return _finite_blob(bench_elsaesser_vars(seed))
