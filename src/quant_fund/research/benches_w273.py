"""Wave-273 compiler-3 benches: optimization passes."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.const_fold import bench_const_fold
from quant_fund.models.inline_expand import bench_inline_expand
from quant_fund.models.loop_unroll import bench_loop_unroll
from quant_fund.models.partial_eval import bench_partial_eval
from quant_fund.models.peephole_opt import bench_peephole_opt
from quant_fund.models.strength_red import bench_strength_red

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


def bench_partial_eval_family(seed: int = _SEED + 1500) -> dict[str, float]:
    return _floats(_finite_blob("partial_eval", bench_partial_eval(seed)))


def bench_peephole_opt_family(seed: int = _SEED + 1501) -> dict[str, float]:
    return _floats(_finite_blob("peephole_opt", bench_peephole_opt(seed)))


def bench_strength_red_family(seed: int = _SEED + 1502) -> dict[str, float]:
    return _floats(_finite_blob("strength_red", bench_strength_red(seed)))


def bench_const_fold_family(seed: int = _SEED + 1503) -> dict[str, float]:
    return _floats(_finite_blob("const_fold", bench_const_fold(seed)))


def bench_loop_unroll_family(seed: int = _SEED + 1504) -> dict[str, float]:
    return _floats(_finite_blob("loop_unroll", bench_loop_unroll(seed)))


def bench_inline_expand_family(seed: int = _SEED + 1505) -> dict[str, float]:
    return _floats(_finite_blob("inline_expand", bench_inline_expand(seed)))
