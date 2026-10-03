"""Wave-405 derived-categories bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bounded_complex import bench_bounded_complex
from quant_fund.models.derived_functor2 import bench_derived_functor2
from quant_fund.models.koszul_dual import bench_koszul_dual
from quant_fund.models.mapping_cone_tri import bench_mapping_cone_tri
from quant_fund.models.t_structure import bench_t_structure
from quant_fund.models.triangulated import bench_triangulated

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


def bench_derived_functor2_family(
    seed: int = _SEED + 2336,
) -> dict[str, float]:
    return _floats(_finite_blob("derived_functor2", bench_derived_functor2(seed)))


def bench_triangulated_family(seed: int = _SEED + 2337) -> dict[str, float]:
    return _floats(_finite_blob("triangulated", bench_triangulated(seed)))


def bench_bounded_complex_family(
    seed: int = _SEED + 2338,
) -> dict[str, float]:
    return _floats(_finite_blob("bounded_complex", bench_bounded_complex(seed)))


def bench_mapping_cone_tri_family(
    seed: int = _SEED + 2339,
) -> dict[str, float]:
    return _floats(_finite_blob("mapping_cone_tri", bench_mapping_cone_tri(seed)))


def bench_koszul_dual_family(seed: int = _SEED + 2340) -> dict[str, float]:
    return _floats(_finite_blob("koszul_dual", bench_koszul_dual(seed)))


def bench_t_structure_family(seed: int = _SEED + 2341) -> dict[str, float]:
    return _floats(_finite_blob("t_structure", bench_t_structure(seed)))
