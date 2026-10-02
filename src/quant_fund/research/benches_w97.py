"""Wave-97 adapters: learning-theory + numerics canon —
proximal-gradient solvers (ISTA/FISTA with restart),
Gauss-Legendre/Hermite + Clenshaw-Curtis + adaptive-Simpson
quadrature, RK4/RK45 + implicit-midpoint + EM/Milstein
ODE/SDE solvers, label-propagation + self-training
semi-supervised learning, OvR/softmax/ECOC multiclass
reductions, and SCoTLASS-style sparse PCA.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.multiclass import bench_multiclass
from quant_fund.models.ode_solvers import bench_ode_solvers
from quant_fund.models.proximal_gradient import bench_proximal_gradient
from quant_fund.models.quadrature import bench_quadrature
from quant_fund.models.semisupervised import bench_semisupervised
from quant_fund.models.sparse_pca import bench_sparse_pca

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


def bench_proximal_gradient_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("proximal_gradient", bench_proximal_gradient(seed=_SEED + 570))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"proximal_gradient bench failed: {exc}") from exc


def bench_quadrature_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("quadrature", bench_quadrature(seed=_SEED + 571)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"quadrature bench failed: {exc}") from exc


def bench_ode_solvers_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("ode_solvers", bench_ode_solvers(seed=_SEED + 572)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ode_solvers bench failed: {exc}") from exc


def bench_semisupervised_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("semisupervised", bench_semisupervised(seed=_SEED + 573))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"semisupervised bench failed: {exc}") from exc


def bench_multiclass_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("multiclass", bench_multiclass(seed=_SEED + 574)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"multiclass bench failed: {exc}") from exc


def bench_sparse_pca_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("sparse_pca", bench_sparse_pca(seed=_SEED + 575)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sparse_pca bench failed: {exc}") from exc
