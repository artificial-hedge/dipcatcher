"""Wave-324 polyhedral-compiler canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.banerjee_dep import bench_banerjee_dep
from quant_fund.models.fourier_motzkin import bench_fourier_motzkin
from quant_fund.models.omega_test import bench_omega_test
from quant_fund.models.pluto_schedule import bench_pluto_schedule
from quant_fund.models.tiling_legality import bench_tiling_legality
from quant_fund.models.vec_legality import bench_vec_legality

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


def bench_fourier_motzkin_family(seed: int = _SEED + 1851) -> dict[str, float]:
    return _floats(_finite_blob("fourier_motzkin", bench_fourier_motzkin(seed)))


def bench_banerjee_dep_family(seed: int = _SEED + 1852) -> dict[str, float]:
    return _floats(_finite_blob("banerjee_dep", bench_banerjee_dep(seed)))


def bench_pluto_schedule_family(seed: int = _SEED + 1853) -> dict[str, float]:
    return _floats(_finite_blob("pluto_schedule", bench_pluto_schedule(seed)))


def bench_tiling_legality_family(seed: int = _SEED + 1854) -> dict[str, float]:
    return _floats(_finite_blob("tiling_legality", bench_tiling_legality(seed)))


def bench_omega_test_family(seed: int = _SEED + 1855) -> dict[str, float]:
    return _floats(_finite_blob("omega_test", bench_omega_test(seed)))


def bench_vec_legality_family(seed: int = _SEED + 1856) -> dict[str, float]:
    return _floats(_finite_blob("vec_legality", bench_vec_legality(seed)))
