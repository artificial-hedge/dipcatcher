"""Wave-1000 elasticity canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.contact_mechanics import bench_contact_mechanics
from quant_fund.models.fracture_mechanics import bench_fracture_mechanics
from quant_fund.models.homogenized_elasticity import bench_homogenized_elasticity
from quant_fund.models.kirchhoff_plate import bench_kirchhoff_plate
from quant_fund.models.mindlin_reissner import bench_mindlin_reissner
from quant_fund.models.navier_elasticity import bench_navier_elasticity

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


def bench_navier_elasticity_family(seed: int = _SEED + 38800) -> dict[str, float]:
    return _finite_blob(bench_navier_elasticity(seed))


def bench_kirchhoff_plate_family(seed: int = _SEED + 38801) -> dict[str, float]:
    return _finite_blob(bench_kirchhoff_plate(seed))


def bench_mindlin_reissner_family(seed: int = _SEED + 38802) -> dict[str, float]:
    return _finite_blob(bench_mindlin_reissner(seed))


def bench_contact_mechanics_family(seed: int = _SEED + 38803) -> dict[str, float]:
    return _finite_blob(bench_contact_mechanics(seed))


def bench_fracture_mechanics_family(seed: int = _SEED + 38804) -> dict[str, float]:
    return _finite_blob(bench_fracture_mechanics(seed))


def bench_homogenized_elasticity_family(seed: int = _SEED + 38805) -> dict[str, float]:
    return _finite_blob(bench_homogenized_elasticity(seed))
