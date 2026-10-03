"""Wave-215 adapters: stochastic-programming canon — two_stage_lshaped,
scenario_tree, saa_consistency, chance_scenario, dro_wasserstein, robust_budget —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chance_scenario import bench_chance_scenario
from quant_fund.models.dro_wasserstein import bench_dro_wasserstein
from quant_fund.models.robust_budget import bench_robust_budget
from quant_fund.models.saa_consistency import bench_saa_consistency
from quant_fund.models.scenario_tree import bench_scenario_tree
from quant_fund.models.two_stage_lshaped import bench_two_stage_lshaped

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


def bench_dro_wasserstein_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dro_wasserstein", bench_dro_wasserstein(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dro_wasserstein bench failed: {exc}") from exc


def bench_two_stage_lshaped_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("two_stage_lshaped", bench_two_stage_lshaped(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"two_stage_lshaped bench failed: {exc}") from exc


def bench_chance_scenario_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("chance_scenario", bench_chance_scenario(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"chance_scenario bench failed: {exc}") from exc


def bench_scenario_tree_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("scenario_tree", bench_scenario_tree(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"scenario_tree bench failed: {exc}") from exc


def bench_saa_consistency_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("saa_consistency", bench_saa_consistency(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"saa_consistency bench failed: {exc}") from exc


def bench_robust_budget_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("robust_budget", bench_robust_budget(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"robust_budget bench failed: {exc}") from exc
