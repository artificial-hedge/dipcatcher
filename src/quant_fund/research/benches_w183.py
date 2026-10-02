"""Wave-126 adapters: exec-summary causal-structure-DL canon — notears,
dagma_lin, golem_ev, notears_mlp, dag_gnn, cam_prune —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cam_prune import bench_cam_prune
from quant_fund.models.dag_gnn import bench_dag_gnn
from quant_fund.models.dagma_lin import bench_dagma_lin
from quant_fund.models.golem_ev import bench_golem_ev
from quant_fund.models.notears import bench_notears
from quant_fund.models.notears_mlp import bench_notears_mlp

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
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_notears_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("notears", bench_notears(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"notears bench failed: {exc}") from exc


def bench_dagma_lin_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dagma_lin", bench_dagma_lin(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dagma_lin bench failed: {exc}") from exc


def bench_golem_ev_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("golem_ev", bench_golem_ev(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"golem_ev bench failed: {exc}") from exc


def bench_notears_mlp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("notears_mlp", bench_notears_mlp(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"notears_mlp bench failed: {exc}") from exc


def bench_dag_gnn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dag_gnn", bench_dag_gnn(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dag_gnn bench failed: {exc}") from exc


def bench_cam_prune_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cam_prune", bench_cam_prune(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cam_prune bench failed: {exc}") from exc
