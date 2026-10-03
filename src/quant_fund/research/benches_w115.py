"""Wave-115 adapters: uncertainty-quantification & surrogate canon —
Smolyak sparse grids, polynomial chaos expansion + Sobol indices,
Bayesian quadrature, Karhunen–Loève expansion, active subspaces,
and multi-index Monte Carlo — each benched on SYNTHETIC problems.
Adapters flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.active_subspace import bench_active_subspace
from quant_fund.models.bayesian_quadrature import bench_bayesian_quadrature
from quant_fund.models.kl_expand import bench_kl_expand
from quant_fund.models.mimc import bench_mimc
from quant_fund.models.pce import bench_pce
from quant_fund.models.smolyak import bench_smolyak

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


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_smolyak_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("smolyak", bench_smolyak(seed=_SEED + 678)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"smolyak bench failed: {exc}") from exc


def bench_pce_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pce", bench_pce(seed=_SEED + 679)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pce bench failed: {exc}") from exc


def bench_bayesian_quadrature_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "bayesian_quadrature",
                bench_bayesian_quadrature(seed=_SEED + 680),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bayesian_quadrature bench failed: {exc}") from exc


def bench_kl_expand_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("kl_expand", bench_kl_expand(seed=_SEED + 681)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"kl_expand bench failed: {exc}") from exc


def bench_active_subspace_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("active_subspace", bench_active_subspace(seed=_SEED + 682)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"active_subspace bench failed: {exc}") from exc


def bench_mimc_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mimc", bench_mimc(seed=_SEED + 683)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mimc bench failed: {exc}") from exc
