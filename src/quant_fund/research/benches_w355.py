"""Wave-355 probability-2 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.conditional_expect import bench_conditional_expect
from quant_fund.models.conv_sum import bench_conv_sum
from quant_fund.models.kolmogorov_axioms import bench_kolmogorov_axioms
from quant_fund.models.markov_ineq import bench_markov_ineq
from quant_fund.models.moment_generating import bench_moment_generating
from quant_fund.models.stochastic_order import bench_stochastic_order

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


def bench_kolmogorov_axioms_family(seed: int = _SEED + 2037) -> dict[str, float]:
    return _floats(_finite_blob("kolmogorov_axioms", bench_kolmogorov_axioms(seed)))


def bench_conditional_expect_family(seed: int = _SEED + 2038) -> dict[str, float]:
    return _floats(_finite_blob("conditional_expect", bench_conditional_expect(seed)))


def bench_markov_ineq_family(seed: int = _SEED + 2039) -> dict[str, float]:
    return _floats(_finite_blob("markov_ineq", bench_markov_ineq(seed)))


def bench_conv_sum_family(seed: int = _SEED + 2040) -> dict[str, float]:
    return _floats(_finite_blob("conv_sum", bench_conv_sum(seed)))


def bench_moment_generating_family(seed: int = _SEED + 2041) -> dict[str, float]:
    return _floats(_finite_blob("moment_generating", bench_moment_generating(seed)))


def bench_stochastic_order_family(seed: int = _SEED + 2042) -> dict[str, float]:
    return _floats(_finite_blob("stochastic_order", bench_stochastic_order(seed)))
