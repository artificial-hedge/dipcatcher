"""Wave-113 adapters: multi-target tracking canon — Jonker–Volgenant
assignment, JPDA single-target-in-clutter, GM-PHD intensity filter,
k-best MHT, covariance-intersection fusion, and Chan's TDOA
multilateration — each benched on SYNTHETIC scenarios. Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cov_int import bench_cov_int
from quant_fund.models.jonker_volgenant import bench_jonker_volgenant
from quant_fund.models.jpda import bench_jpda
from quant_fund.models.mht import bench_mht
from quant_fund.models.phd_filter import bench_phd
from quant_fund.models.tdoa import bench_tdoa

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


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_jonker_volgenant_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("jonker_volgenant", bench_jonker_volgenant(seed=_SEED + 666)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"jonker_volgenant bench failed: {exc}") from exc


def bench_jpda_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("jpda", bench_jpda(seed=_SEED + 667)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"jpda bench failed: {exc}") from exc


def bench_phd_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("phd", bench_phd(seed=_SEED + 668)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"phd bench failed: {exc}") from exc


def bench_mht_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mht", bench_mht(seed=_SEED + 669)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mht bench failed: {exc}") from exc


def bench_cov_int_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cov_int", bench_cov_int(seed=_SEED + 670)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cov_int bench failed: {exc}") from exc


def bench_tdoa_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tdoa", bench_tdoa(seed=_SEED + 671)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tdoa bench failed: {exc}") from exc
