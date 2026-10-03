"""Wave-210 adapters: coding-theory canon — gomory_cut,
column_generation, benders_decomp, lagrangian_relax, branch_and_cut, held_karp —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.benders_decomp import bench_benders_decomp
from quant_fund.models.branch_and_cut import bench_branch_and_cut
from quant_fund.models.column_generation import bench_column_generation
from quant_fund.models.gomory_cut import bench_gomory_cut
from quant_fund.models.held_karp import bench_held_karp
from quant_fund.models.lagrangian_relax import bench_lagrangian_relax

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


def bench_branch_and_cut_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("branch_and_cut", bench_branch_and_cut(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"branch_and_cut bench failed: {exc}") from exc


def bench_gomory_cut_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gomory_cut", bench_gomory_cut(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gomory_cut bench failed: {exc}") from exc


def bench_lagrangian_relax_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lagrangian_relax", bench_lagrangian_relax(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lagrangian_relax bench failed: {exc}") from exc


def bench_column_generation_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("column_generation", bench_column_generation(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"column_generation bench failed: {exc}") from exc


def bench_benders_decomp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("benders_decomp", bench_benders_decomp(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"benders_decomp bench failed: {exc}") from exc


def bench_held_karp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("held_karp", bench_held_karp(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"held_karp bench failed: {exc}") from exc
