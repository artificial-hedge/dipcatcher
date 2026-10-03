"""Wave-126 adapters: exec-summary data-centric canon — active_bald,
data_cartography, el2n_scoring, forgetting_events, influence_func, proto_prune —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.active_bald import bench_active_bald
from quant_fund.models.data_cartography import bench_data_cartography
from quant_fund.models.el2n_scoring import bench_el2n_scoring
from quant_fund.models.forgetting_events import bench_forgetting_events
from quant_fund.models.influence_func import bench_influence_func
from quant_fund.models.proto_prune import bench_proto_prune

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


def bench_active_bald_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("active_bald", bench_active_bald(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"active_bald bench failed: {exc}") from exc


def bench_data_cartography_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("data_cartography", bench_data_cartography(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"data_cartography bench failed: {exc}") from exc


def bench_el2n_scoring_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("el2n_scoring", bench_el2n_scoring(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"el2n_scoring bench failed: {exc}") from exc


def bench_forgetting_events_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("forgetting_events", bench_forgetting_events(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"forgetting_events bench failed: {exc}") from exc


def bench_influence_func_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("influence_func", bench_influence_func(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"influence_func bench failed: {exc}") from exc


def bench_proto_prune_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("proto_prune", bench_proto_prune(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"proto_prune bench failed: {exc}") from exc
