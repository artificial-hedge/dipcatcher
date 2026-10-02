"""Wave-126 adapters: exec-summary graph-exotics canon — algo_reasoning,
pna_agg, virtual_node, gps_transformer, oversmooth_metric, dgn_directional —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.algo_reasoning import bench_algo_reasoning
from quant_fund.models.dgn_directional import bench_dgn_directional
from quant_fund.models.gps_transformer import bench_gps_transformer
from quant_fund.models.oversmooth_metric import bench_oversmooth_metric
from quant_fund.models.pna_agg import bench_pna_agg
from quant_fund.models.virtual_node import bench_virtual_node

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


def bench_algo_reasoning_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("algo_reasoning", bench_algo_reasoning(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"algo_reasoning bench failed: {exc}") from exc


def bench_pna_agg_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pna_agg", bench_pna_agg(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pna_agg bench failed: {exc}") from exc


def bench_virtual_node_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("virtual_node", bench_virtual_node(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"virtual_node bench failed: {exc}") from exc


def bench_gps_transformer_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gps_transformer", bench_gps_transformer(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gps_transformer bench failed: {exc}") from exc


def bench_oversmooth_metric_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("oversmooth_metric", bench_oversmooth_metric(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"oversmooth_metric bench failed: {exc}") from exc


def bench_dgn_directional_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dgn_directional", bench_dgn_directional(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dgn_directional bench failed: {exc}") from exc
