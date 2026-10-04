"""Wave-1032 aerospace-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.aerodynamics import bench_aerodynamics
from quant_fund.models.airfoil_theory import bench_airfoil_theory
from quant_fund.models.flight_dynamics import bench_flight_dynamics
from quant_fund.models.orbital_mechanics2 import bench_orbital_mechanics2
from quant_fund.models.propulsion import bench_propulsion
from quant_fund.models.spacecraft_design import bench_spacecraft_design

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


def bench_aerodynamics_family(seed: int = _SEED + 42000) -> dict[str, float]:
    return _finite_blob(bench_aerodynamics(seed))


def bench_propulsion_family(seed: int = _SEED + 42001) -> dict[str, float]:
    return _finite_blob(bench_propulsion(seed))


def bench_orbital_mechanics2_family(seed: int = _SEED + 42002) -> dict[str, float]:
    return _finite_blob(bench_orbital_mechanics2(seed))


def bench_flight_dynamics_family(seed: int = _SEED + 42003) -> dict[str, float]:
    return _finite_blob(bench_flight_dynamics(seed))


def bench_spacecraft_design_family(seed: int = _SEED + 42004) -> dict[str, float]:
    return _finite_blob(bench_spacecraft_design(seed))


def bench_airfoil_theory_family(seed: int = _SEED + 42005) -> dict[str, float]:
    return _finite_blob(bench_airfoil_theory(seed))
