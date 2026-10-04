"""Wave-383 homotopy-theory-3 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.co_homology import bench_co_homology
from quant_fund.models.em_space import bench_em_space
from quant_fund.models.loop_space import bench_loop_space
from quant_fund.models.mapping_cone import bench_mapping_cone
from quant_fund.models.stiefel_whitney import bench_stiefel_whitney
from quant_fund.models.transfer import bench_transfer

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


def bench_mapping_cone_family(seed: int = _SEED + 2204) -> dict[str, float]:
    return _floats(_finite_blob("mapping_cone", bench_mapping_cone(seed)))


def bench_loop_space_family(seed: int = _SEED + 2205) -> dict[str, float]:
    return _floats(_finite_blob("loop_space", bench_loop_space(seed)))


def bench_em_space_family(seed: int = _SEED + 2206) -> dict[str, float]:
    return _floats(_finite_blob("em_space", bench_em_space(seed)))


def bench_co_homology_family(seed: int = _SEED + 2207) -> dict[str, float]:
    return _floats(_finite_blob("co_homology", bench_co_homology(seed)))


def bench_stiefel_whitney_family(seed: int = _SEED + 2208) -> dict[str, float]:
    return _floats(_finite_blob("stiefel_whitney", bench_stiefel_whitney(seed)))


def bench_transfer_family(seed: int = _SEED + 2209) -> dict[str, float]:
    return _floats(_finite_blob("transfer", bench_transfer(seed)))
