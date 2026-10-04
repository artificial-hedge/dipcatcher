"""Wave-1005 general-relativity canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.einstein_equations import bench_einstein_equations
from quant_fund.models.friedmann_eq import bench_friedmann_eq
from quant_fund.models.gr_birkhoff import bench_gr_birkhoff
from quant_fund.models.kerr_metric import bench_kerr_metric
from quant_fund.models.penrose_diagrams import bench_penrose_diagrams
from quant_fund.models.schwarzschild_metric import bench_schwarzschild_metric

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


def bench_einstein_equations_family(seed: int = _SEED + 39300) -> dict[str, float]:
    return _finite_blob(bench_einstein_equations(seed))


def bench_schwarzschild_metric_family(seed: int = _SEED + 39301) -> dict[str, float]:
    return _finite_blob(bench_schwarzschild_metric(seed))


def bench_friedmann_eq_family(seed: int = _SEED + 39302) -> dict[str, float]:
    return _finite_blob(bench_friedmann_eq(seed))


def bench_kerr_metric_family(seed: int = _SEED + 39303) -> dict[str, float]:
    return _finite_blob(bench_kerr_metric(seed))


def bench_gr_birkhoff_family(seed: int = _SEED + 39304) -> dict[str, float]:
    return _finite_blob(bench_gr_birkhoff(seed))


def bench_penrose_diagrams_family(seed: int = _SEED + 39305) -> dict[str, float]:
    return _finite_blob(bench_penrose_diagrams(seed))
