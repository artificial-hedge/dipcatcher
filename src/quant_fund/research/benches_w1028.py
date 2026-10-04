"""Wave-1028 chemical-engineering canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.fluid_dynamics2 import bench_fluid_dynamics2
from quant_fund.models.heat_exchanger import bench_heat_exchanger
from quant_fund.models.process_control import bench_process_control
from quant_fund.models.reaction_kinetics import bench_reaction_kinetics
from quant_fund.models.separation_proc import bench_separation_proc
from quant_fund.models.thermo_props import bench_thermo_props

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


def bench_reaction_kinetics_family(seed: int = _SEED + 41600) -> dict[str, float]:
    return _finite_blob(bench_reaction_kinetics(seed))


def bench_thermo_props_family(seed: int = _SEED + 41601) -> dict[str, float]:
    return _finite_blob(bench_thermo_props(seed))


def bench_separation_proc_family(seed: int = _SEED + 41602) -> dict[str, float]:
    return _finite_blob(bench_separation_proc(seed))


def bench_heat_exchanger_family(seed: int = _SEED + 41603) -> dict[str, float]:
    return _finite_blob(bench_heat_exchanger(seed))


def bench_fluid_dynamics2_family(seed: int = _SEED + 41604) -> dict[str, float]:
    return _finite_blob(bench_fluid_dynamics2(seed))


def bench_process_control_family(seed: int = _SEED + 41605) -> dict[str, float]:
    return _finite_blob(bench_process_control(seed))
