"""Wave-1009 electrodynamics/optics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dipole_radiation import bench_dipole_radiation
from quant_fund.models.fresnel_eq import bench_fresnel_eq
from quant_fund.models.lorentz_lorenz import bench_lorentz_lorenz
from quant_fund.models.maxwell_equations import bench_maxwell_equations
from quant_fund.models.poynting_vector import bench_poynting_vector
from quant_fund.models.wave_guides import bench_wave_guides

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


def bench_maxwell_equations_family(seed: int = _SEED + 39700) -> dict[str, float]:
    return _finite_blob(bench_maxwell_equations(seed))


def bench_poynting_vector_family(seed: int = _SEED + 39701) -> dict[str, float]:
    return _finite_blob(bench_poynting_vector(seed))


def bench_fresnel_eq_family(seed: int = _SEED + 39702) -> dict[str, float]:
    return _finite_blob(bench_fresnel_eq(seed))


def bench_wave_guides_family(seed: int = _SEED + 39703) -> dict[str, float]:
    return _finite_blob(bench_wave_guides(seed))


def bench_dipole_radiation_family(seed: int = _SEED + 39704) -> dict[str, float]:
    return _finite_blob(bench_dipole_radiation(seed))


def bench_lorentz_lorenz_family(seed: int = _SEED + 39705) -> dict[str, float]:
    return _finite_blob(bench_lorentz_lorenz(seed))
