"""Wave-663 higher-algebra-5 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dunn_additivity import bench_dunn_additivity
from quant_fund.models.e2_algebra import bench_e2_algebra
from quant_fund.models.khovanov_2 import bench_khovanov_2
from quant_fund.models.mckay_correspond import bench_mckay_correspond
from quant_fund.models.swiss_cheese2 import bench_swiss_cheese2
from quant_fund.models.tensor_factorization import (
    bench_tensor_factorization,
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


def bench_e2_algebra_family(
    seed: int = _SEED + 5200,
) -> dict[str, float]:
    return _floats(_finite_blob("e2_algebra", bench_e2_algebra(seed)))


def bench_dunn_additivity_family(
    seed: int = _SEED + 5201,
) -> dict[str, float]:
    return _floats(_finite_blob("dunn_additivity", bench_dunn_additivity(seed)))


def bench_tensor_factorization_family(
    seed: int = _SEED + 5202,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "tensor_factorization",
            bench_tensor_factorization(seed),
        )
    )


def bench_swiss_cheese2_family(
    seed: int = _SEED + 5203,
) -> dict[str, float]:
    return _floats(_finite_blob("swiss_cheese2", bench_swiss_cheese2(seed)))


def bench_mckay_correspond_family(
    seed: int = _SEED + 5204,
) -> dict[str, float]:
    return _floats(_finite_blob("mckay_correspond", bench_mckay_correspond(seed)))


def bench_khovanov_2_family(
    seed: int = _SEED + 5205,
) -> dict[str, float]:
    return _floats(_finite_blob("khovanov_2", bench_khovanov_2(seed)))
