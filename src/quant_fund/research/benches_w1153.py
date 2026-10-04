"""Wave-1153 physical-sciences canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.astrophysics_3 import bench_astrophysics_3
from quant_fund.models.cosmology_3 import bench_cosmology_3
from quant_fund.models.geophysics_3 import bench_geophysics_3
from quant_fund.models.mechanics import bench_mechanics
from quant_fund.models.physics_6 import bench_physics_6
from quant_fund.models.thermodynamics_3 import bench_thermodynamics_3

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


def bench_physics_6_family(seed: int = _SEED + 54100) -> dict[str, float]:
    return _finite_blob(bench_physics_6(seed))


def bench_astrophysics_3_family(seed: int = _SEED + 54101) -> dict[str, float]:
    return _finite_blob(bench_astrophysics_3(seed))


def bench_cosmology_3_family(seed: int = _SEED + 54102) -> dict[str, float]:
    return _finite_blob(bench_cosmology_3(seed))


def bench_geophysics_3_family(seed: int = _SEED + 54103) -> dict[str, float]:
    return _finite_blob(bench_geophysics_3(seed))


def bench_mechanics_family(seed: int = _SEED + 54104) -> dict[str, float]:
    return _finite_blob(bench_mechanics(seed))


def bench_thermodynamics_3_family(seed: int = _SEED + 54105) -> dict[str, float]:
    return _finite_blob(bench_thermodynamics_3(seed))
