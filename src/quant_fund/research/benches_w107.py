"""Wave-107 adapters: transport/HJB PDE canon — Peaceman–Rachford
ADI, Lax–Wendroff advection, WENO5 flux reconstruction,
level-set reinitialization + curvature flow, Sethian fast
marching, and Godunov Burgers — each benched on SYNTHETIC
problems with exact references. Adapters flatten to a finite
float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adi import bench_adi
from quant_fund.models.fast_marching import bench_fast_marching
from quant_fund.models.godunov import bench_godunov
from quant_fund.models.lax_wendroff import bench_lax_wendroff
from quant_fund.models.level_set import bench_level_set
from quant_fund.models.weno import bench_weno

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


def bench_adi_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("adi", bench_adi(seed=_SEED + 630)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"adi bench failed: {exc}") from exc


def bench_lax_wendroff_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("lax_wendroff", bench_lax_wendroff(seed=_SEED + 631))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lax_wendroff bench failed: {exc}") from exc


def bench_weno_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("weno", bench_weno(seed=_SEED + 632)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"weno bench failed: {exc}") from exc


def bench_level_set_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("level_set", bench_level_set(seed=_SEED + 633)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"level_set bench failed: {exc}") from exc


def bench_fast_marching_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("fast_marching", bench_fast_marching(seed=_SEED + 634))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fast_marching bench failed: {exc}") from exc


def bench_godunov_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("godunov", bench_godunov(seed=_SEED + 635)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"godunov bench failed: {exc}") from exc
