"""Wave-126 adapters: exec-summary certified + constrained robustness canon — randomized_smoothing,
ibp_bounds, crown_bound, lipschitz_net, vector_neurons, gumbel_topk —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.crown_bound import bench_crown_bound
from quant_fund.models.gumbel_topk import bench_gumbel_topk
from quant_fund.models.ibp_bounds import bench_ibp_bounds
from quant_fund.models.lipschitz_net import bench_lipschitz_net
from quant_fund.models.randomized_smoothing import bench_randomized_smoothing
from quant_fund.models.vector_neurons import bench_vector_neurons

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


def bench_randomized_smoothing_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("randomized_smoothing", bench_randomized_smoothing(seed=_SEED + 834))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"randomized_smoothing bench failed: {exc}") from exc


def bench_ibp_bounds_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ibp_bounds", bench_ibp_bounds(seed=_SEED + 835)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ibp_bounds bench failed: {exc}") from exc


def bench_crown_bound_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("crown_bound", bench_crown_bound(seed=_SEED + 836)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"crown_bound bench failed: {exc}") from exc


def bench_lipschitz_net_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lipschitz_net", bench_lipschitz_net(seed=_SEED + 837)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lipschitz_net bench failed: {exc}") from exc


def bench_vector_neurons_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vector_neurons", bench_vector_neurons(seed=_SEED + 838)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vector_neurons bench failed: {exc}") from exc


def bench_gumbel_topk_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gumbel_topk", bench_gumbel_topk(seed=_SEED + 839)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gumbel_topk bench failed: {exc}") from exc
