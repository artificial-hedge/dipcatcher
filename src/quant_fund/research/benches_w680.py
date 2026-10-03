"""Wave-680 spectral-AG-6 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.spectral_abelian import bench_spectral_abelian
from quant_fund.models.spectral_crystal import bench_spectral_crystal
from quant_fund.models.spectral_etale2 import bench_spectral_etale2
from quant_fund.models.spectral_perfect import bench_spectral_perfect
from quant_fund.models.spectral_proper import bench_spectral_proper
from quant_fund.models.spectral_smooth2 import bench_spectral_smooth2

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


def bench_spectral_perfect_family(
    seed: int = _SEED + 6900,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_perfect", bench_spectral_perfect(seed)))


def bench_spectral_smooth2_family(
    seed: int = _SEED + 6901,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_smooth2", bench_spectral_smooth2(seed)))


def bench_spectral_etale2_family(
    seed: int = _SEED + 6902,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_etale2", bench_spectral_etale2(seed)))


def bench_spectral_abelian_family(
    seed: int = _SEED + 6903,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_abelian", bench_spectral_abelian(seed)))


def bench_spectral_crystal_family(
    seed: int = _SEED + 6904,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_crystal", bench_spectral_crystal(seed)))


def bench_spectral_proper_family(
    seed: int = _SEED + 6905,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_proper", bench_spectral_proper(seed)))
