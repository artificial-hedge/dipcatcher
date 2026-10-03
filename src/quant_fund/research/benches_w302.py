"""Wave-302 compiler-5/JIT canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.card_table_gc import bench_card_table_gc
from quant_fund.models.escape_analysis import bench_escape_analysis
from quant_fund.models.gvn_pre import bench_gvn_pre
from quant_fund.models.osr_deopt import bench_osr_deopt
from quant_fund.models.ssa_repair import bench_ssa_repair
from quant_fund.models.trace_tree import bench_trace_tree

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


def bench_card_table_gc_family(seed: int = _SEED + 1718) -> dict[str, float]:
    return _floats(_finite_blob("card_table_gc", bench_card_table_gc(seed)))


def bench_escape_analysis_family(seed: int = _SEED + 1719) -> dict[str, float]:
    return _floats(_finite_blob("escape_analysis", bench_escape_analysis(seed)))


def bench_osr_deopt_family(seed: int = _SEED + 1720) -> dict[str, float]:
    return _floats(_finite_blob("osr_deopt", bench_osr_deopt(seed)))


def bench_trace_tree_family(seed: int = _SEED + 1721) -> dict[str, float]:
    return _floats(_finite_blob("trace_tree", bench_trace_tree(seed)))


def bench_ssa_repair_family(seed: int = _SEED + 1722) -> dict[str, float]:
    return _floats(_finite_blob("ssa_repair", bench_ssa_repair(seed)))


def bench_gvn_pre_family(seed: int = _SEED + 1723) -> dict[str, float]:
    return _floats(_finite_blob("gvn_pre", bench_gvn_pre(seed)))
