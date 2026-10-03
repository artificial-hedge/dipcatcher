"""Wave-452 geometric-Langlands bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.d_module import bench_d_module
from quant_fund.models.geometric_langlands import bench_geometric_langlands
from quant_fund.models.hecke_eig import bench_hecke_eig
from quant_fund.models.kernel_fun import bench_kernel_fun
from quant_fund.models.opers_g import bench_opers_g
from quant_fund.models.ramified_l import bench_ramified_l

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


def bench_d_module_family(seed: int = _SEED + 2618) -> dict[str, float]:
    return _floats(_finite_blob("d_module", bench_d_module(seed)))


def bench_geometric_langlands_family(seed: int = _SEED + 2619) -> dict[str, float]:
    return _floats(_finite_blob("geometric_langlands", bench_geometric_langlands(seed)))


def bench_hecke_eig_family(seed: int = _SEED + 2620) -> dict[str, float]:
    return _floats(_finite_blob("hecke_eig", bench_hecke_eig(seed)))


def bench_opers_g_family(seed: int = _SEED + 2621) -> dict[str, float]:
    return _floats(_finite_blob("opers_g", bench_opers_g(seed)))


def bench_ramified_l_family(seed: int = _SEED + 2622) -> dict[str, float]:
    return _floats(_finite_blob("ramified_l", bench_ramified_l(seed)))


def bench_kernel_fun_family(seed: int = _SEED + 2623) -> dict[str, float]:
    return _floats(_finite_blob("kernel_fun", bench_kernel_fun(seed)))
