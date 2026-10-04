"""Wave-771 copula bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.clayton_copula import (
    bench_clayton_copula,
)
from quant_fund.models.copula_gauss import bench_copula_gauss
from quant_fund.models.copula_t import bench_copula_t
from quant_fund.models.frank_copula import bench_frank_copula
from quant_fund.models.gumbel_copula import bench_gumbel_copula
from quant_fund.models.joe_copula import bench_joe_copula

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


def bench_copula_gauss_family(
    seed: int = _SEED + 16000,
) -> dict[str, float]:
    return _floats(_finite_blob("copula_gauss", bench_copula_gauss(seed)))


def bench_copula_t_family(
    seed: int = _SEED + 16001,
) -> dict[str, float]:
    return _floats(_finite_blob("copula_t", bench_copula_t(seed)))


def bench_clayton_copula_family(
    seed: int = _SEED + 16002,
) -> dict[str, float]:
    return _floats(_finite_blob("clayton_copula", bench_clayton_copula(seed)))


def bench_gumbel_copula_family(
    seed: int = _SEED + 16003,
) -> dict[str, float]:
    return _floats(_finite_blob("gumbel_copula", bench_gumbel_copula(seed)))


def bench_frank_copula_family(
    seed: int = _SEED + 16004,
) -> dict[str, float]:
    return _floats(_finite_blob("frank_copula", bench_frank_copula(seed)))


def bench_joe_copula_family(
    seed: int = _SEED + 16005,
) -> dict[str, float]:
    return _floats(_finite_blob("joe_copula", bench_joe_copula(seed)))
