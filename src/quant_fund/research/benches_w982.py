"""Wave-982 potential-theory-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.boundary_regular import bench_boundary_regular
from quant_fund.models.capacitary_pot import bench_capacitary_pot
from quant_fund.models.dirichlet_problem import bench_dirichlet_problem
from quant_fund.models.energy_principle import bench_energy_principle
from quant_fund.models.equilibrium_measure import bench_equilibrium_measure
from quant_fund.models.thin_set import bench_thin_set

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


def bench_dirichlet_problem_family(seed: int = _SEED + 37000) -> dict[str, float]:
    return _finite_blob(bench_dirichlet_problem(seed))


def bench_energy_principle_family(seed: int = _SEED + 37001) -> dict[str, float]:
    return _finite_blob(bench_energy_principle(seed))


def bench_equilibrium_measure_family(seed: int = _SEED + 37002) -> dict[str, float]:
    return _finite_blob(bench_equilibrium_measure(seed))


def bench_thin_set_family(seed: int = _SEED + 37003) -> dict[str, float]:
    return _finite_blob(bench_thin_set(seed))


def bench_boundary_regular_family(seed: int = _SEED + 37004) -> dict[str, float]:
    return _finite_blob(bench_boundary_regular(seed))


def bench_capacitary_pot_family(seed: int = _SEED + 37005) -> dict[str, float]:
    return _finite_blob(bench_capacitary_pot(seed))
