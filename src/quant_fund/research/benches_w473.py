"""Wave-473 higher-algebra-3 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cartier_mod import bench_cartier_mod
from quant_fund.models.crystalline_stack import bench_crystalline_stack
from quant_fund.models.cyclotomic2 import bench_cyclotomic2
from quant_fund.models.thh_2 import bench_thh_2
from quant_fund.models.trt_functor import bench_trt_functor
from quant_fund.models.witt_vec2 import bench_witt_vec2

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


def bench_thh_2_family(seed: int = _SEED + 2744) -> dict[str, float]:
    return _floats(_finite_blob("thh_2", bench_thh_2(seed)))


def bench_cyclotomic2_family(seed: int = _SEED + 2745) -> dict[str, float]:
    return _floats(_finite_blob("cyclotomic2", bench_cyclotomic2(seed)))


def bench_cartier_mod_family(seed: int = _SEED + 2746) -> dict[str, float]:
    return _floats(_finite_blob("cartier_mod", bench_cartier_mod(seed)))


def bench_witt_vec2_family(seed: int = _SEED + 2747) -> dict[str, float]:
    return _floats(_finite_blob("witt_vec2", bench_witt_vec2(seed)))


def bench_crystalline_stack_family(seed: int = _SEED + 2748) -> dict[str, float]:
    return _floats(_finite_blob("crystalline_stack", bench_crystalline_stack(seed)))


def bench_trt_functor_family(seed: int = _SEED + 2749) -> dict[str, float]:
    return _floats(_finite_blob("trt_functor", bench_trt_functor(seed)))
