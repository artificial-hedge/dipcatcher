"""Wave-565 arithmetic-statistics bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bhargava_lic import bench_bhargava_lic
from quant_fund.models.cohen_lenstra import bench_cohen_lenstra
from quant_fund.models.elliptic_rank import bench_elliptic_rank
from quant_fund.models.malle_conj import bench_malle_conj
from quant_fund.models.prime_gaps import bench_prime_gaps
from quant_fund.models.zhang_maynard import bench_zhang_maynard

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


def bench_bhargava_lic_family(seed: int = _SEED + 3296) -> dict[str, float]:
    return _floats(_finite_blob("bhargava_lic", bench_bhargava_lic(seed)))


def bench_cohen_lenstra_family(seed: int = _SEED + 3297) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "cohen_lenstra",
            bench_cohen_lenstra(seed),
        )
    )


def bench_elliptic_rank_family(seed: int = _SEED + 3298) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "elliptic_rank",
            bench_elliptic_rank(seed),
        )
    )


def bench_malle_conj_family(seed: int = _SEED + 3299) -> dict[str, float]:
    return _floats(_finite_blob("malle_conj", bench_malle_conj(seed)))


def bench_prime_gaps_family(seed: int = _SEED + 3300) -> dict[str, float]:
    return _floats(_finite_blob("prime_gaps", bench_prime_gaps(seed)))


def bench_zhang_maynard_family(seed: int = _SEED + 3301) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "zhang_maynard",
            bench_zhang_maynard(seed),
        )
    )
