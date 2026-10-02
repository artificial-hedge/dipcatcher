"""Wave-126 adapters: exec-summary geometric + structured net canon — hyperbolic_nn,
capsule_dynamic, siren_inr, equivar_gnn, monotonic_net, sort_net —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.capsule_dynamic import bench_capsule_dynamic
from quant_fund.models.equivar_gnn import bench_equivar_gnn
from quant_fund.models.hyperbolic_nn import bench_hyperbolic_nn
from quant_fund.models.monotonic_net import bench_monotonic_net
from quant_fund.models.siren_inr import bench_siren_inr
from quant_fund.models.sort_net import bench_sort_net

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


def bench_hyperbolic_nn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hyperbolic_nn", bench_hyperbolic_nn(seed=_SEED + 828)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hyperbolic_nn bench failed: {exc}") from exc


def bench_capsule_dynamic_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("capsule_dynamic", bench_capsule_dynamic(seed=_SEED + 829)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"capsule_dynamic bench failed: {exc}") from exc


def bench_siren_inr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("siren_inr", bench_siren_inr(seed=_SEED + 830)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"siren_inr bench failed: {exc}") from exc


def bench_equivar_gnn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("equivar_gnn", bench_equivar_gnn(seed=_SEED + 831)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"equivar_gnn bench failed: {exc}") from exc


def bench_monotonic_net_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("monotonic_net", bench_monotonic_net(seed=_SEED + 832)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"monotonic_net bench failed: {exc}") from exc


def bench_sort_net_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sort_net", bench_sort_net(seed=_SEED + 833)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sort_net bench failed: {exc}") from exc
