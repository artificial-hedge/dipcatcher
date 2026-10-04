"""Wave-1047 marine-biology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.aquaculture import bench_aquaculture
from quant_fund.models.benthic_biology import bench_benthic_biology
from quant_fund.models.coral_reef_ecology import bench_coral_reef_ecology
from quant_fund.models.fisheries_science import bench_fisheries_science
from quant_fund.models.marine_ecology import bench_marine_ecology
from quant_fund.models.plankton_dynamics import bench_plankton_dynamics

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


def bench_plankton_dynamics_family(seed: int = _SEED + 43500) -> dict[str, float]:
    return _finite_blob(bench_plankton_dynamics(seed))


def bench_marine_ecology_family(seed: int = _SEED + 43501) -> dict[str, float]:
    return _finite_blob(bench_marine_ecology(seed))


def bench_fisheries_science_family(seed: int = _SEED + 43502) -> dict[str, float]:
    return _finite_blob(bench_fisheries_science(seed))


def bench_aquaculture_family(seed: int = _SEED + 43503) -> dict[str, float]:
    return _finite_blob(bench_aquaculture(seed))


def bench_benthic_biology_family(seed: int = _SEED + 43504) -> dict[str, float]:
    return _finite_blob(bench_benthic_biology(seed))


def bench_coral_reef_ecology_family(seed: int = _SEED + 43505) -> dict[str, float]:
    return _finite_blob(bench_coral_reef_ecology(seed))
