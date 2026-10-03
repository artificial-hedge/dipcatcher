"""Wave-121 adapters: best-arm identification canon — lil_ucb,
sequential_halving, median_elim, ugape, ttts, track_stop — each
benched on SYNTHETIC Gaussian bandits. Adapters flatten to a finite
float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.lil_ucb import bench_lil_ucb
from quant_fund.models.median_elim import bench_median_elim
from quant_fund.models.sequential_halving import bench_sequential_halving
from quant_fund.models.track_stop import bench_track_stop
from quant_fund.models.ttts import bench_ttts
from quant_fund.models.ugape import bench_ugape

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


def bench_lil_ucb_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lil_ucb", bench_lil_ucb(seed=_SEED + 714)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lil_ucb bench failed: {exc}") from exc


def bench_sequential_halving_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("sequential_halving", bench_sequential_halving(seed=_SEED + 715))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sequential_halving bench failed: {exc}") from exc


def bench_median_elim_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("median_elim", bench_median_elim(seed=_SEED + 716)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"median_elim bench failed: {exc}") from exc


def bench_ugape_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ugape", bench_ugape(seed=_SEED + 717)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ugape bench failed: {exc}") from exc


def bench_ttts_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ttts", bench_ttts(seed=_SEED + 718)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ttts bench failed: {exc}") from exc


def bench_track_stop_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("track_stop", bench_track_stop(seed=_SEED + 719)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"track_stop bench failed: {exc}") from exc
