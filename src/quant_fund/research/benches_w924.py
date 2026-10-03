"""Wave-924 Bayesian-nonparametrics-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.beta_bernoulli import bench_beta_bernoulli
from quant_fund.models.crp_table import bench_crp_table
from quant_fund.models.dp_mm import bench_dp_mm
from quant_fund.models.gem_distribution import bench_gem_distribution
from quant_fund.models.gibbs_type import bench_gibbs_type
from quant_fund.models.neutral_process import bench_neutral_process

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


def bench_gem_distribution_family(seed: int = _SEED + 31200) -> dict[str, float]:
    return _finite_blob(bench_gem_distribution(seed))


def bench_dp_mm_family(seed: int = _SEED + 31201) -> dict[str, float]:
    return _finite_blob(bench_dp_mm(seed))


def bench_crp_table_family(seed: int = _SEED + 31202) -> dict[str, float]:
    return _finite_blob(bench_crp_table(seed))


def bench_beta_bernoulli_family(seed: int = _SEED + 31203) -> dict[str, float]:
    return _finite_blob(bench_beta_bernoulli(seed))


def bench_neutral_process_family(seed: int = _SEED + 31204) -> dict[str, float]:
    return _finite_blob(bench_neutral_process(seed))


def bench_gibbs_type_family(seed: int = _SEED + 31205) -> dict[str, float]:
    return _finite_blob(bench_gibbs_type(seed))
