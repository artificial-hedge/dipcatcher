"""Wave-590 algebraic-K-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bloch_k import bench_bloch_k
from quant_fund.models.gersten_ss import bench_gersten_ss
from quant_fund.models.loday_k import bench_loday_k
from quant_fund.models.quillen_plus import bench_quillen_plus
from quant_fund.models.suslin_k import bench_suslin_k
from quant_fund.models.volodin_k import bench_volodin_k

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


def bench_quillen_plus_family(
    seed: int = _SEED + 3446,
) -> dict[str, float]:
    return _floats(_finite_blob("quillen_plus", bench_quillen_plus(seed)))


def bench_gersten_ss_family(
    seed: int = _SEED + 3447,
) -> dict[str, float]:
    return _floats(_finite_blob("gersten_ss", bench_gersten_ss(seed)))


def bench_loday_k_family(seed: int = _SEED + 3448) -> dict[str, float]:
    return _floats(_finite_blob("loday_k", bench_loday_k(seed)))


def bench_volodin_k_family(
    seed: int = _SEED + 3449,
) -> dict[str, float]:
    return _floats(_finite_blob("volodin_k", bench_volodin_k(seed)))


def bench_suslin_k_family(seed: int = _SEED + 3450) -> dict[str, float]:
    return _floats(_finite_blob("suslin_k", bench_suslin_k(seed)))


def bench_bloch_k_family(seed: int = _SEED + 3451) -> dict[str, float]:
    return _floats(_finite_blob("bloch_k", bench_bloch_k(seed)))
