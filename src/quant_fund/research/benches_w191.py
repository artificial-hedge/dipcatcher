"""Wave-126 adapters: exec-summary RL-exploration canon — ride_explore,
count_bonus, ngu_explore, rnd_explore, icm_explore, go_explore —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.count_bonus import bench_count_bonus
from quant_fund.models.go_explore import bench_go_explore
from quant_fund.models.icm_explore import bench_icm_explore
from quant_fund.models.ngu_explore import bench_ngu_explore
from quant_fund.models.ride_explore import bench_ride_explore
from quant_fund.models.rnd_explore import bench_rnd_explore

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


def bench_ride_explore_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ride_explore", bench_ride_explore(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ride_explore bench failed: {exc}") from exc


def bench_count_bonus_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("count_bonus", bench_count_bonus(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"count_bonus bench failed: {exc}") from exc


def bench_ngu_explore_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ngu_explore", bench_ngu_explore(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ngu_explore bench failed: {exc}") from exc


def bench_rnd_explore_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rnd_explore", bench_rnd_explore(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rnd_explore bench failed: {exc}") from exc


def bench_icm_explore_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("icm_explore", bench_icm_explore(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"icm_explore bench failed: {exc}") from exc


def bench_go_explore_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("go_explore", bench_go_explore(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"go_explore bench failed: {exc}") from exc
