"""Wave-264 numerical-linalg-3 canon adapter: SYNTHETIC benches."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.block_lanczos import bench_block_lanczos
from quant_fund.models.divide_conquer_eig import bench_divide_conquer_eig
from quant_fund.models.dqds import bench_dqds
from quant_fund.models.fgmres import bench_fgmres
from quant_fund.models.randomized_qb import bench_randomized_qb
from quant_fund.models.sparse_cholesky import bench_sparse_cholesky

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


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


def bench_divide_conquer_eig_family(seed: int = _SEED + 1410) -> dict[str, float]:
    return bench_divide_conquer_eig(seed)


def bench_dqds_family(seed: int = _SEED + 1411) -> dict[str, float]:
    return bench_dqds(seed)


def bench_block_lanczos_family(seed: int = _SEED + 1412) -> dict[str, float]:
    return bench_block_lanczos(seed)


def bench_randomized_qb_family(seed: int = _SEED + 1413) -> dict[str, float]:
    return bench_randomized_qb(seed)


def bench_sparse_cholesky_family(seed: int = _SEED + 1414) -> dict[str, float]:
    return bench_sparse_cholesky(seed)


def bench_fgmres_family(seed: int = _SEED + 1415) -> dict[str, float]:
    return bench_fgmres(seed)
