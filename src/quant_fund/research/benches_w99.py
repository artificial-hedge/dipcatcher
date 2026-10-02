"""Wave-99 adapters: Bayesian-computation + density + latent +
multi-task canon — conjugate Gibbs samplers, Laplace
approximation with damped Newton, Gaussian KDE with LOO-CV
bandwidth, whitened tensor power iteration, Dirichlet-
evidence (subjective-logic) classification, and mean-
shrinkage multi-task ridge.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.evidential import bench_evidential
from quant_fund.models.gibbs_sampler import bench_gibbs_sampler
from quant_fund.models.kde import bench_kde
from quant_fund.models.laplace_approx import bench_laplace_approx
from quant_fund.models.multi_task import bench_multi_task
from quant_fund.models.tensor_power import bench_tensor_power

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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
                flat[f"{k}_{i}"] = f
    return flat


def _isinstance_floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_gibbs_sampler_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("gibbs_sampler", bench_gibbs_sampler(seed=_SEED + 582))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gibbs_sampler bench failed: {exc}") from exc


def bench_laplace_approx_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("laplace_approx", bench_laplace_approx(seed=_SEED + 583))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"laplace_approx bench failed: {exc}") from exc


def bench_kde_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("kde", bench_kde(seed=_SEED + 584)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"kde bench failed: {exc}") from exc


def bench_tensor_power_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("tensor_power", bench_tensor_power(seed=_SEED + 585))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tensor_power bench failed: {exc}") from exc


def bench_evidential_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("evidential", bench_evidential(seed=_SEED + 586)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"evidential bench failed: {exc}") from exc


def bench_multi_task_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("multi_task", bench_multi_task(seed=_SEED + 587)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"multi_task bench failed: {exc}") from exc
