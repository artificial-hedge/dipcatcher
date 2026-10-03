"""Wave-106 adapters: stiff/multistep time-integration canon —
BDF1/BDF2, Adams–Bashforth + PECE, Radau IIA, Strang splitting,
ETDRK4, and Crank–Nicolson heat stepping — each benched on
SYNTHETIC problems with exact solutions so convergence orders
are measurable. Adapters flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adams import bench_adams
from quant_fund.models.bdf import bench_bdf
from quant_fund.models.crank_nicolson import bench_crank_nicolson
from quant_fund.models.etdrk4 import bench_etdrk4
from quant_fund.models.radau import bench_radau
from quant_fund.models.strang import bench_strang

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


def bench_bdf_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("bdf", bench_bdf(seed=_SEED + 624)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bdf bench failed: {exc}") from exc


def bench_adams_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("adams", bench_adams(seed=_SEED + 625)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"adams bench failed: {exc}") from exc


def bench_radau_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("radau", bench_radau(seed=_SEED + 626)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"radau bench failed: {exc}") from exc


def bench_strang_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("strang", bench_strang(seed=_SEED + 627)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"strang bench failed: {exc}") from exc


def bench_etdrk4_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("etdrk4", bench_etdrk4(seed=_SEED + 628)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"etdrk4 bench failed: {exc}") from exc


def bench_crank_nicolson_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("crank_nicolson", bench_crank_nicolson(seed=_SEED + 629))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"crank_nicolson bench failed: {exc}") from exc
