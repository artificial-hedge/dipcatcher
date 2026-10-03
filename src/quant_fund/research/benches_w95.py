"""Wave-95 adapters: classic ML canon III — Bayesian
linear + ARD evidence, Bayes-net structure/inference,
linear+nonlinear conjugate gradient, Frank-Wolfe /
pairwise-FW, OMP+K-SVD sparse coding, (μ/μ,λ) and (1+1)
evolution strategies.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bayesian_linear import bench_bayesian_linear
from quant_fund.models.conjugate_gradient import bench_conjugate_gradient
from quant_fund.models.evolution_strategies import bench_evolution_strategies
from quant_fund.models.frank_wolfe import bench_frank_wolfe
from quant_fund.models.graphical_models import bench_graphical_models
from quant_fund.models.sparse_coding import bench_sparse_coding

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError(f"non-finite {k} in {name}")
        flat[k] = float(arr.ravel()[0])
    if not flat:
        raise ValueError(f"{name} returned no metrics")
    return flat


def _isinstance_floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_bayesian_linear_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("bayesian_linear", bench_bayesian_linear(seed=_SEED + 558))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bayesian_linear bench failed: {exc}") from exc


def bench_graphical_models_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("graphical_models", bench_graphical_models(seed=_SEED + 559))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"graphical_models bench failed: {exc}") from exc


def bench_conjugate_gradient_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("conjugate_gradient", bench_conjugate_gradient(seed=_SEED + 560))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"conjugate_gradient bench failed: {exc}") from exc


def bench_frank_wolfe_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("frank_wolfe", bench_frank_wolfe(seed=_SEED + 561)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"frank_wolfe bench failed: {exc}") from exc


def bench_sparse_coding_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("sparse_coding", bench_sparse_coding(seed=_SEED + 562))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sparse_coding bench failed: {exc}") from exc


def bench_evolution_strategies_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("evolution_strategies", bench_evolution_strategies(seed=_SEED + 563))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"evolution_strategies bench failed: {exc}") from exc
