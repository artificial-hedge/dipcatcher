"""Wave-126 adapters: exec-summary scheduling-canon — tsp_branchbound,
johnson_flowshop, knapsack_dp, neh_heuristic, lpt_schedule, spt_weighted —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.johnson_flowshop import bench_johnson_flowshop
from quant_fund.models.knapsack_dp import bench_knapsack_dp
from quant_fund.models.lpt_schedule import bench_lpt_schedule
from quant_fund.models.neh_heuristic import bench_neh_heuristic
from quant_fund.models.spt_weighted import bench_spt_weighted
from quant_fund.models.tsp_branchbound import bench_tsp_branchbound

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


def bench_tsp_branchbound_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tsp_branchbound", bench_tsp_branchbound(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tsp_branchbound bench failed: {exc}") from exc


def bench_johnson_flowshop_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("johnson_flowshop", bench_johnson_flowshop(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"johnson_flowshop bench failed: {exc}") from exc


def bench_knapsack_dp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("knapsack_dp", bench_knapsack_dp(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"knapsack_dp bench failed: {exc}") from exc


def bench_neh_heuristic_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("neh_heuristic", bench_neh_heuristic(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"neh_heuristic bench failed: {exc}") from exc


def bench_lpt_schedule_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lpt_schedule", bench_lpt_schedule(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lpt_schedule bench failed: {exc}") from exc


def bench_spt_weighted_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("spt_weighted", bench_spt_weighted(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"spt_weighted bench failed: {exc}") from exc
