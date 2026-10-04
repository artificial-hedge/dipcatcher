"""Wave-785 semimartingale-2 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dolean_mart import bench_dolean_mart
from quant_fund.models.follmer_mart import (
    bench_follmer_mart,
)
from quant_fund.models.local_mart2 import bench_local_mart2
from quant_fund.models.protter_ito import bench_protter_ito
from quant_fund.models.strong_sol import bench_strong_sol
from quant_fund.models.usual_cond import bench_usual_cond

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


def bench_usual_cond_family(
    seed: int = _SEED + 17400,
) -> dict[str, float]:
    return _floats(_finite_blob("usual_cond", bench_usual_cond(seed)))


def bench_dolean_mart_family(
    seed: int = _SEED + 17401,
) -> dict[str, float]:
    return _floats(_finite_blob("dolean_mart", bench_dolean_mart(seed)))


def bench_strong_sol_family(
    seed: int = _SEED + 17402,
) -> dict[str, float]:
    return _floats(_finite_blob("strong_sol", bench_strong_sol(seed)))


def bench_local_mart2_family(
    seed: int = _SEED + 17403,
) -> dict[str, float]:
    return _floats(_finite_blob("local_mart2", bench_local_mart2(seed)))


def bench_follmer_mart_family(
    seed: int = _SEED + 17404,
) -> dict[str, float]:
    return _floats(_finite_blob("follmer_mart", bench_follmer_mart(seed)))


def bench_protter_ito_family(
    seed: int = _SEED + 17405,
) -> dict[str, float]:
    return _floats(_finite_blob("protter_ito", bench_protter_ito(seed)))
