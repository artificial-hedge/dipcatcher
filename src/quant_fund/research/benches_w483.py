"""Wave-483 homotopy-10 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bar_spec import bench_bar_spec
from quant_fund.models.dyer_lashof import bench_dyer_lashof
from quant_fund.models.free_loop import bench_free_loop
from quant_fund.models.loop_functor import bench_loop_functor
from quant_fund.models.steenrod_ops import bench_steenrod_ops
from quant_fund.models.sullivan_min import bench_sullivan_min

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


def bench_steenrod_ops_family(seed: int = _SEED + 2804) -> dict[str, float]:
    return _floats(_finite_blob("steenrod_ops", bench_steenrod_ops(seed)))


def bench_dyer_lashof_family(seed: int = _SEED + 2805) -> dict[str, float]:
    return _floats(_finite_blob("dyer_lashof", bench_dyer_lashof(seed)))


def bench_bar_spec_family(seed: int = _SEED + 2806) -> dict[str, float]:
    return _floats(_finite_blob("bar_spec", bench_bar_spec(seed)))


def bench_free_loop_family(seed: int = _SEED + 2807) -> dict[str, float]:
    return _floats(_finite_blob("free_loop", bench_free_loop(seed)))


def bench_sullivan_min_family(seed: int = _SEED + 2808) -> dict[str, float]:
    return _floats(_finite_blob("sullivan_min", bench_sullivan_min(seed)))


def bench_loop_functor_family(seed: int = _SEED + 2809) -> dict[str, float]:
    return _floats(_finite_blob("loop_functor", bench_loop_functor(seed)))
