"""Wave-432 spectral-sequences-3 adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.atiyah_hirzebruch import (
    bench_atiyah_hirzebruch,
)
from quant_fund.models.descent_ss import bench_descent_ss
from quant_fund.models.leary_ss import bench_leary_ss
from quant_fund.models.motivic_ss import bench_motivic_ss
from quant_fund.models.serre_ss3 import bench_serre_ss3
from quant_fund.models.vanishing_ss import bench_vanishing_ss

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


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


def bench_atiyah_hirzebruch_family(
    seed: int = _SEED + 2498,
) -> dict[str, float]:
    return _floats(_finite_blob("atiyah_hirzebruch", bench_atiyah_hirzebruch(seed)))


def bench_serre_ss3_family(
    seed: int = _SEED + 2499,
) -> dict[str, float]:
    return _floats(_finite_blob("serre_ss3", bench_serre_ss3(seed)))


def bench_leary_ss_family(
    seed: int = _SEED + 2500,
) -> dict[str, float]:
    return _floats(_finite_blob("leary_ss", bench_leary_ss(seed)))


def bench_descent_ss_family(
    seed: int = _SEED + 2501,
) -> dict[str, float]:
    return _floats(_finite_blob("descent_ss", bench_descent_ss(seed)))


def bench_motivic_ss_family(
    seed: int = _SEED + 2502,
) -> dict[str, float]:
    return _floats(_finite_blob("motivic_ss", bench_motivic_ss(seed)))


def bench_vanishing_ss_family(
    seed: int = _SEED + 2503,
) -> dict[str, float]:
    return _floats(_finite_blob("vanishing_ss", bench_vanishing_ss(seed)))
