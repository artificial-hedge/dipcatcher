"""Wave-1114 physics-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.classical_mechanics import bench_classical_mechanics
from quant_fund.models.condensed_matter_2 import bench_condensed_matter_2
from quant_fund.models.nuclear_physics import bench_nuclear_physics
from quant_fund.models.plasma_physics import bench_plasma_physics
from quant_fund.models.quantum_mechanics_2 import bench_quantum_mechanics_2
from quant_fund.models.statistical_mechanics_2 import bench_statistical_mechanics_2

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


def bench_classical_mechanics_family(seed: int = _SEED + 50200) -> dict[str, float]:
    return _finite_blob(bench_classical_mechanics(seed))


def bench_quantum_mechanics_2_family(seed: int = _SEED + 50201) -> dict[str, float]:
    return _finite_blob(bench_quantum_mechanics_2(seed))


def bench_statistical_mechanics_2_family(seed: int = _SEED + 50202) -> dict[str, float]:
    return _finite_blob(bench_statistical_mechanics_2(seed))


def bench_nuclear_physics_family(seed: int = _SEED + 50203) -> dict[str, float]:
    return _finite_blob(bench_nuclear_physics(seed))


def bench_plasma_physics_family(seed: int = _SEED + 50204) -> dict[str, float]:
    return _finite_blob(bench_plasma_physics(seed))


def bench_condensed_matter_2_family(seed: int = _SEED + 50205) -> dict[str, float]:
    return _finite_blob(bench_condensed_matter_2(seed))
