"""Wave-750 dimer/Ising bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chelkak_ising import bench_chelkak_ising
from quant_fund.models.duminil_copin2 import (
    bench_duminil_copin2,
)
from quant_fund.models.hongler_ising import bench_hongler_ising
from quant_fund.models.kenyon_dimers import bench_kenyon_dimers
from quant_fund.models.smirnov_ising import bench_smirnov_ising
from quant_fund.models.thurston_tiling import (
    bench_thurston_tiling,
)

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


def bench_smirnov_ising_family(
    seed: int = _SEED + 13900,
) -> dict[str, float]:
    return _floats(_finite_blob("smirnov_ising", bench_smirnov_ising(seed)))


def bench_chelkak_ising_family(
    seed: int = _SEED + 13901,
) -> dict[str, float]:
    return _floats(_finite_blob("chelkak_ising", bench_chelkak_ising(seed)))


def bench_kenyon_dimers_family(
    seed: int = _SEED + 13902,
) -> dict[str, float]:
    return _floats(_finite_blob("kenyon_dimers", bench_kenyon_dimers(seed)))


def bench_thurston_tiling_family(
    seed: int = _SEED + 13903,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "thurston_tiling",
            bench_thurston_tiling(seed),
        )
    )


def bench_duminil_copin2_family(
    seed: int = _SEED + 13904,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "duminil_copin2",
            bench_duminil_copin2(seed),
        )
    )


def bench_hongler_ising_family(
    seed: int = _SEED + 13905,
) -> dict[str, float]:
    return _floats(_finite_blob("hongler_ising", bench_hongler_ising(seed)))
