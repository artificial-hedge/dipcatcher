"""Wave-770 extreme-value bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.frechet_domain import bench_frechet_domain
from quant_fund.models.gumbel_domain import bench_gumbel_domain
from quant_fund.models.hill_est import bench_hill_est
from quant_fund.models.peak_over import bench_peak_over
from quant_fund.models.pickands_est import bench_pickands_est
from quant_fund.models.weibull_domain import (
    bench_weibull_domain,
)

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_gumbel_domain_family(
    seed: int = _SEED + 15900,
) -> dict[str, float]:
    return _floats(_finite_blob("gumbel_domain", bench_gumbel_domain(seed)))


def bench_weibull_domain_family(
    seed: int = _SEED + 15901,
) -> dict[str, float]:
    return _floats(_finite_blob("weibull_domain", bench_weibull_domain(seed)))


def bench_frechet_domain_family(
    seed: int = _SEED + 15902,
) -> dict[str, float]:
    return _floats(_finite_blob("frechet_domain", bench_frechet_domain(seed)))


def bench_peak_over_family(
    seed: int = _SEED + 15903,
) -> dict[str, float]:
    return _floats(_finite_blob("peak_over", bench_peak_over(seed)))


def bench_hill_est_family(
    seed: int = _SEED + 15904,
) -> dict[str, float]:
    return _floats(_finite_blob("hill_est", bench_hill_est(seed)))


def bench_pickands_est_family(
    seed: int = _SEED + 15905,
) -> dict[str, float]:
    return _floats(_finite_blob("pickands_est", bench_pickands_est(seed)))
