"""Wave-485 p-adic-4 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.banach_colmez2 import bench_banach_colmez2
from quant_fund.models.bc_space import bench_bc_space
from quant_fund.models.fargues_curve2 import bench_fargues_curve2
from quant_fund.models.local_shimura import bench_local_shimura
from quant_fund.models.lubin_tate2 import bench_lubin_tate2
from quant_fund.models.scholze_weinstein import bench_scholze_weinstein

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


def bench_lubin_tate2_family(seed: int = _SEED + 2816) -> dict[str, float]:
    return _floats(_finite_blob("lubin_tate2", bench_lubin_tate2(seed)))


def bench_bc_space_family(seed: int = _SEED + 2817) -> dict[str, float]:
    return _floats(_finite_blob("bc_space", bench_bc_space(seed)))


def bench_local_shimura_family(seed: int = _SEED + 2818) -> dict[str, float]:
    return _floats(_finite_blob("local_shimura", bench_local_shimura(seed)))


def bench_scholze_weinstein_family(seed: int = _SEED + 2819) -> dict[str, float]:
    return _floats(_finite_blob("scholze_weinstein", bench_scholze_weinstein(seed)))


def bench_fargues_curve2_family(seed: int = _SEED + 2820) -> dict[str, float]:
    return _floats(_finite_blob("fargues_curve2", bench_fargues_curve2(seed)))


def bench_banach_colmez2_family(seed: int = _SEED + 2821) -> dict[str, float]:
    return _floats(_finite_blob("banach_colmez2", bench_banach_colmez2(seed)))
