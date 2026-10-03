"""Wave-109 adapters: motion-planning canon — Dubins shortest
paths, RRT/RRT*, probabilistic roadmap, dynamic-window approach,
minimum-snap trajectories, Frenet optimal planning — each
benched on SYNTHETIC scenarios with known references.
Adapters flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dubins import bench_dubins
from quant_fund.models.dwa import bench_dwa
from quant_fund.models.frenet import bench_frenet
from quant_fund.models.min_snap import bench_min_snap
from quant_fund.models.prm import bench_prm
from quant_fund.models.rrt import bench_rrt

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
                flat[f"{k}_{i}"] = f
    return flat


def _isinstance_floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_dubins_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("dubins", bench_dubins(seed=_SEED + 642)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dubins bench failed: {exc}") from exc


def bench_rrt_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("rrt", bench_rrt(seed=_SEED + 643)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rrt bench failed: {exc}") from exc


def bench_prm_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("prm", bench_prm(seed=_SEED + 644)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"prm bench failed: {exc}") from exc


def bench_dwa_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("dwa", bench_dwa(seed=_SEED + 645)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dwa bench failed: {exc}") from exc


def bench_min_snap_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("min_snap", bench_min_snap(seed=_SEED + 646)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"min_snap bench failed: {exc}") from exc


def bench_frenet_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("frenet", bench_frenet(seed=_SEED + 647)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"frenet bench failed: {exc}") from exc
