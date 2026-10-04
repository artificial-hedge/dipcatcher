"""Wave-1154 fundamental-physics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.electromagnetism import bench_electromagnetism
from quant_fund.models.nuclear_physics_2 import bench_nuclear_physics_2
from quant_fund.models.optics_4 import bench_optics_4
from quant_fund.models.particle_physics import bench_particle_physics
from quant_fund.models.quantum_physics import bench_quantum_physics
from quant_fund.models.relativity_3 import bench_relativity_3

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


def bench_electromagnetism_family(seed: int = _SEED + 54200) -> dict[str, float]:
    return _finite_blob(bench_electromagnetism(seed))


def bench_optics_4_family(seed: int = _SEED + 54201) -> dict[str, float]:
    return _finite_blob(bench_optics_4(seed))


def bench_nuclear_physics_2_family(seed: int = _SEED + 54202) -> dict[str, float]:
    return _finite_blob(bench_nuclear_physics_2(seed))


def bench_particle_physics_family(seed: int = _SEED + 54203) -> dict[str, float]:
    return _finite_blob(bench_particle_physics(seed))


def bench_quantum_physics_family(seed: int = _SEED + 54204) -> dict[str, float]:
    return _finite_blob(bench_quantum_physics(seed))


def bench_relativity_3_family(seed: int = _SEED + 54205) -> dict[str, float]:
    return _finite_blob(bench_relativity_3(seed))
