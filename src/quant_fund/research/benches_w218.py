"""Wave-218 adapters: advanced-derivatives canon — dupire_localvol,
sabr_calib, deep_hedge, heston_calib, barrier_adjoint, andreasen_huge —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.andreasen_huge import bench_andreasen_huge
from quant_fund.models.barrier_adjoint import bench_barrier_adjoint
from quant_fund.models.deep_hedge import bench_deep_hedge
from quant_fund.models.dupire_localvol import bench_dupire_localvol
from quant_fund.models.heston_calib import bench_heston_calib
from quant_fund.models.sabr_calib import bench_sabr_calib

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


def bench_barrier_adjoint_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("barrier_adjoint", bench_barrier_adjoint(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"barrier_adjoint bench failed: {exc}") from exc


def bench_dupire_localvol_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dupire_localvol", bench_dupire_localvol(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dupire_localvol bench failed: {exc}") from exc


def bench_heston_calib_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("heston_calib", bench_heston_calib(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"heston_calib bench failed: {exc}") from exc


def bench_sabr_calib_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sabr_calib", bench_sabr_calib(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sabr_calib bench failed: {exc}") from exc


def bench_deep_hedge_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("deep_hedge", bench_deep_hedge(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"deep_hedge bench failed: {exc}") from exc


def bench_andreasen_huge_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("andreasen_huge", bench_andreasen_huge(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"andreasen_huge bench failed: {exc}") from exc
