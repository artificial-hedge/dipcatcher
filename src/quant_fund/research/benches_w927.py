"""Wave-927 information-geometry-3 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.ebanch_diverge import bench_ebanch_diverge
from quant_fund.models.expectation_param import bench_expectation_param
from quant_fund.models.fisher_metric2 import bench_fisher_metric2
from quant_fund.models.potential_fn import bench_potential_fn
from quant_fund.models.renyi_div import bench_renyi_div
from quant_fund.models.shannon_gibbs import bench_shannon_gibbs

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


def bench_fisher_metric2_family(seed: int = _SEED + 31500) -> dict[str, float]:
    return _finite_blob(bench_fisher_metric2(seed))


def bench_expectation_param_family(seed: int = _SEED + 31501) -> dict[str, float]:
    return _finite_blob(bench_expectation_param(seed))


def bench_potential_fn_family(seed: int = _SEED + 31502) -> dict[str, float]:
    return _finite_blob(bench_potential_fn(seed))


def bench_ebanch_diverge_family(seed: int = _SEED + 31503) -> dict[str, float]:
    return _finite_blob(bench_ebanch_diverge(seed))


def bench_shannon_gibbs_family(seed: int = _SEED + 31504) -> dict[str, float]:
    return _finite_blob(bench_shannon_gibbs(seed))


def bench_renyi_div_family(seed: int = _SEED + 31505) -> dict[str, float]:
    return _finite_blob(bench_renyi_div(seed))
