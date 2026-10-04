"""Wave-1021 epidemiology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.branching_epidemic import bench_branching_epidemic
from quant_fund.models.herd_immunity import bench_herd_immunity
from quant_fund.models.r0_estimation import bench_r0_estimation
from quant_fund.models.seir_epidemic import bench_seir_epidemic
from quant_fund.models.sir_epidemic import bench_sir_epidemic
from quant_fund.models.sis_epidemic import bench_sis_epidemic

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


def bench_sir_epidemic_family(seed: int = _SEED + 40900) -> dict[str, float]:
    return _finite_blob(bench_sir_epidemic(seed))


def bench_sis_epidemic_family(seed: int = _SEED + 40901) -> dict[str, float]:
    return _finite_blob(bench_sis_epidemic(seed))


def bench_seir_epidemic_family(seed: int = _SEED + 40902) -> dict[str, float]:
    return _finite_blob(bench_seir_epidemic(seed))


def bench_r0_estimation_family(seed: int = _SEED + 40903) -> dict[str, float]:
    return _finite_blob(bench_r0_estimation(seed))


def bench_herd_immunity_family(seed: int = _SEED + 40904) -> dict[str, float]:
    return _finite_blob(bench_herd_immunity(seed))


def bench_branching_epidemic_family(seed: int = _SEED + 40905) -> dict[str, float]:
    return _finite_blob(bench_branching_epidemic(seed))
