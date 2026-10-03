"""Wave-409 homotopy-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cohend import bench_cohend
from quant_fund.models.dold_kan import bench_dold_kan
from quant_fund.models.eilenberg_zilber import bench_eilenberg_zilber
from quant_fund.models.postnikov import bench_postnikov
from quant_fund.models.spectral_seq2 import bench_spectral_seq2
from quant_fund.models.stable_range import bench_stable_range

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


def bench_spectral_seq2_family(
    seed: int = _SEED + 2360,
) -> dict[str, float]:
    return _floats(_finite_blob("spectral_seq2", bench_spectral_seq2(seed)))


def bench_eilenberg_zilber_family(
    seed: int = _SEED + 2361,
) -> dict[str, float]:
    return _floats(_finite_blob("eilenberg_zilber", bench_eilenberg_zilber(seed)))


def bench_dold_kan_family(seed: int = _SEED + 2362) -> dict[str, float]:
    return _floats(_finite_blob("dold_kan", bench_dold_kan(seed)))


def bench_postnikov_family(seed: int = _SEED + 2363) -> dict[str, float]:
    return _floats(_finite_blob("postnikov", bench_postnikov(seed)))


def bench_stable_range_family(
    seed: int = _SEED + 2364,
) -> dict[str, float]:
    return _floats(_finite_blob("stable_range", bench_stable_range(seed)))


def bench_cohend_family(seed: int = _SEED + 2365) -> dict[str, float]:
    return _floats(_finite_blob("cohend", bench_cohend(seed)))
