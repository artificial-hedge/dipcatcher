"""Wave-1008 thermodynamics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.carnot_cycle import bench_carnot_cycle
from quant_fund.models.critical_phenomena import bench_critical_phenomena
from quant_fund.models.entropy_production import bench_entropy_production
from quant_fund.models.fluctuation_dissipation import bench_fluctuation_dissipation
from quant_fund.models.maxwell_relations import bench_maxwell_relations
from quant_fund.models.phase_transitions import bench_phase_transitions

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


def bench_carnot_cycle_family(seed: int = _SEED + 39600) -> dict[str, float]:
    return _finite_blob(bench_carnot_cycle(seed))


def bench_maxwell_relations_family(seed: int = _SEED + 39601) -> dict[str, float]:
    return _finite_blob(bench_maxwell_relations(seed))


def bench_phase_transitions_family(seed: int = _SEED + 39602) -> dict[str, float]:
    return _finite_blob(bench_phase_transitions(seed))


def bench_critical_phenomena_family(seed: int = _SEED + 39603) -> dict[str, float]:
    return _finite_blob(bench_critical_phenomena(seed))


def bench_fluctuation_dissipation_family(seed: int = _SEED + 39604) -> dict[str, float]:
    return _finite_blob(bench_fluctuation_dissipation(seed))


def bench_entropy_production_family(seed: int = _SEED + 39605) -> dict[str, float]:
    return _finite_blob(bench_entropy_production(seed))
