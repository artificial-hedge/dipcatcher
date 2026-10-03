"""Wave-499 Hodge-theory bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.hodge_decomp import bench_hodge_decomp
from quant_fund.models.l2_hodge import bench_l2_hodge
from quant_fund.models.limit_mhs import bench_limit_mhs
from quant_fund.models.mixed_hodge import bench_mixed_hodge
from quant_fund.models.period_map import bench_period_map
from quant_fund.models.vhs_polarized import bench_vhs_polarized

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


def bench_hodge_decomp_family(seed: int = _SEED + 2900) -> dict[str, float]:
    return _floats(_finite_blob("hodge_decomp", bench_hodge_decomp(seed)))


def bench_l2_hodge_family(seed: int = _SEED + 2901) -> dict[str, float]:
    return _floats(_finite_blob("l2_hodge", bench_l2_hodge(seed)))


def bench_mixed_hodge_family(seed: int = _SEED + 2902) -> dict[str, float]:
    return _floats(_finite_blob("mixed_hodge", bench_mixed_hodge(seed)))


def bench_period_map_family(seed: int = _SEED + 2903) -> dict[str, float]:
    return _floats(_finite_blob("period_map", bench_period_map(seed)))


def bench_vhs_polarized_family(seed: int = _SEED + 2904) -> dict[str, float]:
    return _floats(_finite_blob("vhs_polarized", bench_vhs_polarized(seed)))


def bench_limit_mhs_family(seed: int = _SEED + 2905) -> dict[str, float]:
    return _floats(_finite_blob("limit_mhs", bench_limit_mhs(seed)))
