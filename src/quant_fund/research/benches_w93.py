"""Wave-93 adapters: classic-ML canon — CART/RF/GBM tree
ensembles, NCA/LMNN metric learning, SVDD/Mahalanobis/LOF
one-class classification, Laplace GP classification,
GS/Wirtinger phase retrieval, factorization machines.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.factorization_machine import (
    bench_factorization_machine as _fm_bench,
)
from quant_fund.models.gp_classification import (
    bench_gp_classification as _gpc_bench,
)
from quant_fund.models.metric_learning import (
    bench_metric_learning as _ml_bench,
)
from quant_fund.models.one_class_classification import (
    bench_one_class as _oc_bench,
)
from quant_fund.models.phase_retrieval import (
    bench_phase_retrieval as _pr_bench,
)
from quant_fund.models.tree_ensembles import bench_trees

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


def bench_tree_ensembles() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("tree_ensembles", bench_trees(seed=_SEED + 546)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tree_ensembles bench failed: {exc}") from exc


def bench_metric_learning() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("metric_learning", _ml_bench(seed=_SEED + 547)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"metric_learning bench failed: {exc}") from exc


def bench_one_class_classification() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("one_class_classification", _oc_bench(seed=_SEED + 548))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"one_class_classification bench failed: {exc}") from exc


def bench_gp_classification() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("gp_classification", _gpc_bench(seed=_SEED + 549)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gp_classification bench failed: {exc}") from exc


def bench_phase_retrieval() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("phase_retrieval", _pr_bench(seed=_SEED + 550)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"phase_retrieval bench failed: {exc}") from exc


def bench_factorization_machine() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("factorization_machine", _fm_bench(seed=_SEED + 551))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"factorization_machine bench failed: {exc}") from exc
