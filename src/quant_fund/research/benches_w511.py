"""Wave-511 Fargues-Scholze bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bun_g import bench_bun_g
from quant_fund.models.fs_diamond import bench_fs_diamond
from quant_fund.models.geometric_satake import bench_geometric_satake
from quant_fund.models.hecke_stack import bench_hecke_stack
from quant_fund.models.v_sheaf import bench_v_sheaf
from quant_fund.models.y_diamond import bench_y_diamond

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


def bench_fs_diamond_family(seed: int = _SEED + 2972) -> dict[str, float]:
    return _floats(_finite_blob("fs_diamond", bench_fs_diamond(seed)))


def bench_geometric_satake_family(seed: int = _SEED + 2973) -> dict[str, float]:
    return _floats(_finite_blob("geometric_satake", bench_geometric_satake(seed)))


def bench_v_sheaf_family(seed: int = _SEED + 2974) -> dict[str, float]:
    return _floats(_finite_blob("v_sheaf", bench_v_sheaf(seed)))


def bench_bun_g_family(seed: int = _SEED + 2975) -> dict[str, float]:
    return _floats(_finite_blob("bun_g", bench_bun_g(seed)))


def bench_hecke_stack_family(seed: int = _SEED + 2976) -> dict[str, float]:
    return _floats(_finite_blob("hecke_stack", bench_hecke_stack(seed)))


def bench_y_diamond_family(seed: int = _SEED + 2977) -> dict[str, float]:
    return _floats(_finite_blob("y_diamond", bench_y_diamond(seed)))
