"""Wave-294 compiler-4 canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bb_reorder import bench_bb_reorder
from quant_fund.models.cfg_simplify import bench_cfg_simplify
from quant_fund.models.jump_thread import bench_jump_thread
from quant_fund.models.modulo_sched import bench_modulo_sched
from quant_fund.models.tail_dup import bench_tail_dup
from quant_fund.models.tree_cover import bench_tree_cover

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


def bench_tree_cover_family(seed: int = _SEED + 1670) -> dict[str, float]:
    return _floats(_finite_blob("tree_cover", bench_tree_cover(seed)))


def bench_modulo_sched_family(seed: int = _SEED + 1671) -> dict[str, float]:
    return _floats(_finite_blob("modulo_sched", bench_modulo_sched(seed)))


def bench_jump_thread_family(seed: int = _SEED + 1672) -> dict[str, float]:
    return _floats(_finite_blob("jump_thread", bench_jump_thread(seed)))


def bench_tail_dup_family(seed: int = _SEED + 1673) -> dict[str, float]:
    return _floats(_finite_blob("tail_dup", bench_tail_dup(seed)))


def bench_cfg_simplify_family(seed: int = _SEED + 1674) -> dict[str, float]:
    return _floats(_finite_blob("cfg_simplify", bench_cfg_simplify(seed)))


def bench_bb_reorder_family(seed: int = _SEED + 1675) -> dict[str, float]:
    return _floats(_finite_blob("bb_reorder", bench_bb_reorder(seed)))
