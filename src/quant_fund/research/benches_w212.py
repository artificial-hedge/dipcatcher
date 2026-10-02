"""Wave-212 adapters: approximation-algorithm canon — greedy_set_cover,
primal_dual_vc, lp_rounding_sc, fptas_knapsack, local_search_maxcut, christofides_tsp —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.christofides_tsp import bench_christofides_tsp
from quant_fund.models.fptas_knapsack import bench_fptas_knapsack
from quant_fund.models.greedy_set_cover import bench_greedy_set_cover
from quant_fund.models.local_search_maxcut import bench_local_search_maxcut
from quant_fund.models.lp_rounding_sc import bench_lp_rounding_sc
from quant_fund.models.primal_dual_vc import bench_primal_dual_vc

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


def bench_local_search_maxcut_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("local_search_maxcut", bench_local_search_maxcut(seed=_SEED + 960))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"local_search_maxcut bench failed: {exc}") from exc


def bench_greedy_set_cover_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("greedy_set_cover", bench_greedy_set_cover(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"greedy_set_cover bench failed: {exc}") from exc


def bench_fptas_knapsack_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fptas_knapsack", bench_fptas_knapsack(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fptas_knapsack bench failed: {exc}") from exc


def bench_primal_dual_vc_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("primal_dual_vc", bench_primal_dual_vc(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"primal_dual_vc bench failed: {exc}") from exc


def bench_lp_rounding_sc_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lp_rounding_sc", bench_lp_rounding_sc(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lp_rounding_sc bench failed: {exc}") from exc


def bench_christofides_tsp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("christofides_tsp", bench_christofides_tsp(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"christofides_tsp bench failed: {exc}") from exc
