"""Wave-999 GMT-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.brakke_varifolds import bench_brakke_varifolds
from quant_fund.models.currents_theory import bench_currents_theory
from quant_fund.models.flat_chains import bench_flat_chains
from quant_fund.models.integral_currents import bench_integral_currents
from quant_fund.models.rectifiable_measures import bench_rectifiable_measures
from quant_fund.models.varifold_theory import bench_varifold_theory

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


def bench_currents_theory_family(seed: int = _SEED + 38700) -> dict[str, float]:
    return _finite_blob(bench_currents_theory(seed))


def bench_varifold_theory_family(seed: int = _SEED + 38701) -> dict[str, float]:
    return _finite_blob(bench_varifold_theory(seed))


def bench_flat_chains_family(seed: int = _SEED + 38702) -> dict[str, float]:
    return _finite_blob(bench_flat_chains(seed))


def bench_integral_currents_family(seed: int = _SEED + 38703) -> dict[str, float]:
    return _finite_blob(bench_integral_currents(seed))


def bench_rectifiable_measures_family(seed: int = _SEED + 38704) -> dict[str, float]:
    return _finite_blob(bench_rectifiable_measures(seed))


def bench_brakke_varifolds_family(seed: int = _SEED + 38705) -> dict[str, float]:
    return _finite_blob(bench_brakke_varifolds(seed))
