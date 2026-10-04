"""Wave-617 algebraic-K-6 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.berrick_k import bench_berrick_k
from quant_fund.models.gersen_suslin import bench_gersen_suslin
from quant_fund.models.gillet_thomason import (
    bench_gillet_thomason,
)
from quant_fund.models.hermitian_quillen import (
    bench_hermitian_quillen,
)
from quant_fund.models.k_theory4 import bench_k_theory4
from quant_fund.models.khomo_k import bench_khomo_k

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


def bench_gillet_thomason_family(
    seed: int = _SEED + 3608,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gillet_thomason",
            bench_gillet_thomason(seed),
        )
    )


def bench_khomo_k_family(seed: int = _SEED + 3609) -> dict[str, float]:
    return _floats(_finite_blob("khomo_k", bench_khomo_k(seed)))


def bench_k_theory4_family(
    seed: int = _SEED + 3610,
) -> dict[str, float]:
    return _floats(_finite_blob("k_theory4", bench_k_theory4(seed)))


def bench_gersen_suslin_family(
    seed: int = _SEED + 3611,
) -> dict[str, float]:
    return _floats(_finite_blob("gersen_suslin", bench_gersen_suslin(seed)))


def bench_berrick_k_family(
    seed: int = _SEED + 3612,
) -> dict[str, float]:
    return _floats(_finite_blob("berrick_k", bench_berrick_k(seed)))


def bench_hermitian_quillen_family(
    seed: int = _SEED + 3613,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "hermitian_quillen",
            bench_hermitian_quillen(seed),
        )
    )
