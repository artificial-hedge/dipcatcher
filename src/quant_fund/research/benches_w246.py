"""Wave-246 adapters: language-runtime canon — bytecode VM,
threaded dispatch, closure conversion, trampoline TCO, inline
cache, tagged pointers — SYNTHETIC correctness benches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bytecode_vm import bench_bytecode_vm
from quant_fund.models.closure_conv import bench_closure_conv
from quant_fund.models.inline_cache import bench_inline_cache
from quant_fund.models.nan_tagging import bench_nan_tagging
from quant_fund.models.tail_call_tramp import bench_tail_call_tramp
from quant_fund.models.threaded_interp import bench_threaded_interp

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


def bench_bytecode_vm_family(seed: int = _SEED + 1230) -> dict[str, float]:
    return bench_bytecode_vm(seed)


def bench_closure_conv_family(seed: int = _SEED + 1231) -> dict[str, float]:
    return bench_closure_conv(seed)


def bench_inline_cache_family(seed: int = _SEED + 1232) -> dict[str, float]:
    return bench_inline_cache(seed)


def bench_nan_tagging_family(seed: int = _SEED + 1233) -> dict[str, float]:
    return bench_nan_tagging(seed)


def bench_tail_call_tramp_family(seed: int = _SEED + 1234) -> dict[str, float]:
    return bench_tail_call_tramp(seed)


def bench_threaded_interp_family(seed: int = _SEED + 1235) -> dict[str, float]:
    return bench_threaded_interp(seed)
