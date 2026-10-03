"""Wave-239 adapters: programming-languages canon — HM inference,
tree-walk interpreter, CPS transform, macro expander, GC, STLC —
SYNTHETIC correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cps_transform import bench_cps_transform
from quant_fund.models.gc_marksweep import bench_gc_marksweep
from quant_fund.models.hm_inference import bench_hm_inference
from quant_fund.models.macro_expand import bench_macro_expand
from quant_fund.models.simple_types import bench_simple_types
from quant_fund.models.tree_walk_interp import bench_tree_walk_interp

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


def bench_cps_transform_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cps_transform", bench_cps_transform(seed=_SEED + 1160)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cps_transform bench failed: {exc}") from exc


def bench_gc_marksweep_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gc_marksweep", bench_gc_marksweep(seed=_SEED + 1161)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gc_marksweep bench failed: {exc}") from exc


def bench_hm_inference_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hm_inference", bench_hm_inference(seed=_SEED + 1162)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hm_inference bench failed: {exc}") from exc


def bench_macro_expand_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("macro_expand", bench_macro_expand(seed=_SEED + 1163)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"macro_expand bench failed: {exc}") from exc


def bench_simple_types_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("simple_types", bench_simple_types(seed=_SEED + 1164)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"simple_types bench failed: {exc}") from exc


def bench_tree_walk_interp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("tree_walk_interp", bench_tree_walk_interp(seed=_SEED + 1165)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tree_walk_interp bench failed: {exc}") from exc
