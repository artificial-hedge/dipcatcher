"""Wave-993 nonlinear-functional-analysis canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.degree_theory import bench_degree_theory
from quant_fund.models.krein_rutman import bench_krein_rutman
from quant_fund.models.maximal_monotone import bench_maximal_monotone
from quant_fund.models.minty_browder import bench_minty_browder
from quant_fund.models.monotone_op import bench_monotone_op
from quant_fund.models.schauder_fixed import bench_schauder_fixed

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


def bench_monotone_op_family(seed: int = _SEED + 38100) -> dict[str, float]:
    return _finite_blob(bench_monotone_op(seed))


def bench_degree_theory_family(seed: int = _SEED + 38101) -> dict[str, float]:
    return _finite_blob(bench_degree_theory(seed))


def bench_schauder_fixed_family(seed: int = _SEED + 38102) -> dict[str, float]:
    return _finite_blob(bench_schauder_fixed(seed))


def bench_krein_rutman_family(seed: int = _SEED + 38103) -> dict[str, float]:
    return _finite_blob(bench_krein_rutman(seed))


def bench_minty_browder_family(seed: int = _SEED + 38104) -> dict[str, float]:
    return _finite_blob(bench_minty_browder(seed))


def bench_maximal_monotone_family(seed: int = _SEED + 38105) -> dict[str, float]:
    return _finite_blob(bench_maximal_monotone(seed))
