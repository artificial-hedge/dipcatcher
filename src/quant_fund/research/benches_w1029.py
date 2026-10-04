"""Wave-1029 mechanical-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.fatigue_life import bench_fatigue_life
from quant_fund.models.kinematics import bench_kinematics
from quant_fund.models.machine_design import bench_machine_design
from quant_fund.models.solid_mechanics import bench_solid_mechanics
from quant_fund.models.tribology import bench_tribology
from quant_fund.models.vibration_analysis import bench_vibration_analysis

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


def bench_solid_mechanics_family(seed: int = _SEED + 41700) -> dict[str, float]:
    return _finite_blob(bench_solid_mechanics(seed))


def bench_vibration_analysis_family(seed: int = _SEED + 41701) -> dict[str, float]:
    return _finite_blob(bench_vibration_analysis(seed))


def bench_fatigue_life_family(seed: int = _SEED + 41702) -> dict[str, float]:
    return _finite_blob(bench_fatigue_life(seed))


def bench_tribology_family(seed: int = _SEED + 41703) -> dict[str, float]:
    return _finite_blob(bench_tribology(seed))


def bench_machine_design_family(seed: int = _SEED + 41704) -> dict[str, float]:
    return _finite_blob(bench_machine_design(seed))


def bench_kinematics_family(seed: int = _SEED + 41705) -> dict[str, float]:
    return _finite_blob(bench_kinematics(seed))
