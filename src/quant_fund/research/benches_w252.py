"""Wave-252 adapters: interpreters-3 canon — generational GC,
Cheney semi-space copying, vtable dispatch, polymorphic inline
caches, ANF conversion, trampolined tail calls — SYNTHETIC benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.anf_cps import bench_anf_cps
from quant_fund.models.compacting_gc import bench_compacting_gc
from quant_fund.models.dispatch_table import bench_dispatch_table
from quant_fund.models.gen_gc import bench_gen_gc
from quant_fund.models.poly_inline_cache import bench_poly_inline_cache
from quant_fund.models.trampoline_tc import bench_trampoline_tc

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


def bench_gen_gc_family(seed: int = _SEED + 1290) -> dict[str, float]:
    return bench_gen_gc(seed)


def bench_compacting_gc_family(seed: int = _SEED + 1291) -> dict[str, float]:
    return bench_compacting_gc(seed)


def bench_dispatch_table_family(seed: int = _SEED + 1292) -> dict[str, float]:
    return bench_dispatch_table(seed)


def bench_poly_inline_cache_family(seed: int = _SEED + 1293) -> dict[str, float]:
    return bench_poly_inline_cache(seed)


def bench_anf_cps_family(seed: int = _SEED + 1294) -> dict[str, float]:
    return bench_anf_cps(seed)


def bench_trampoline_tc_family(seed: int = _SEED + 1295) -> dict[str, float]:
    return bench_trampoline_tc(seed)
