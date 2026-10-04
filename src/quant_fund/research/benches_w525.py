"""Wave-525 ergodic-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bernoulli_shift import bench_bernoulli_shift
from quant_fund.models.birkhoff import bench_birkhoff
from quant_fund.models.entropy_ks import bench_entropy_ks
from quant_fund.models.mean_ergodic import bench_mean_ergodic
from quant_fund.models.mixing_weak import bench_mixing_weak
from quant_fund.models.osceledets import bench_osceledets

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


def bench_birkhoff_family(seed: int = _SEED + 3056) -> dict[str, float]:
    return _floats(_finite_blob("birkhoff", bench_birkhoff(seed)))


def bench_mean_ergodic_family(seed: int = _SEED + 3057) -> dict[str, float]:
    return _floats(_finite_blob("mean_ergodic", bench_mean_ergodic(seed)))


def bench_mixing_weak_family(seed: int = _SEED + 3058) -> dict[str, float]:
    return _floats(_finite_blob("mixing_weak", bench_mixing_weak(seed)))


def bench_entropy_ks_family(seed: int = _SEED + 3059) -> dict[str, float]:
    return _floats(_finite_blob("entropy_ks", bench_entropy_ks(seed)))


def bench_bernoulli_shift_family(
    seed: int = _SEED + 3060,
) -> dict[str, float]:
    return _floats(_finite_blob("bernoulli_shift", bench_bernoulli_shift(seed)))


def bench_osceledets_family(seed: int = _SEED + 3061) -> dict[str, float]:
    return _floats(_finite_blob("osceledets", bench_osceledets(seed)))
