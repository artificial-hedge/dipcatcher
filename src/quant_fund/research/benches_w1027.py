"""Wave-1027 materials-science canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.ceramics import bench_ceramics
from quant_fund.models.crystal_structure import bench_crystal_structure
from quant_fund.models.metallurgy import bench_metallurgy
from quant_fund.models.nanomaterials import bench_nanomaterials
from quant_fund.models.polymer_physics import bench_polymer_physics
from quant_fund.models.superconductivity import bench_superconductivity

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


def bench_crystal_structure_family(seed: int = _SEED + 41500) -> dict[str, float]:
    return _finite_blob(bench_crystal_structure(seed))


def bench_polymer_physics_family(seed: int = _SEED + 41501) -> dict[str, float]:
    return _finite_blob(bench_polymer_physics(seed))


def bench_metallurgy_family(seed: int = _SEED + 41502) -> dict[str, float]:
    return _finite_blob(bench_metallurgy(seed))


def bench_ceramics_family(seed: int = _SEED + 41503) -> dict[str, float]:
    return _finite_blob(bench_ceramics(seed))


def bench_nanomaterials_family(seed: int = _SEED + 41504) -> dict[str, float]:
    return _finite_blob(bench_nanomaterials(seed))


def bench_superconductivity_family(seed: int = _SEED + 41505) -> dict[str, float]:
    return _finite_blob(bench_superconductivity(seed))
