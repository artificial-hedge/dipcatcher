"""Wave-233 adapters: compiler-2 canon — SSA construction, SCCP,
GVN, register coalescing, list scheduling, LICM — SYNTHETIC program
optimization benches vs sequential-execution oracles.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gvn_elim import bench_gvn_elim
from quant_fund.models.instr_sched import bench_instr_sched
from quant_fund.models.licm_hoist import bench_licm_hoist
from quant_fund.models.reg_coalesce import bench_reg_coalesce
from quant_fund.models.sccp_const import bench_sccp_const
from quant_fund.models.ssa_construct import bench_ssa_construct

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


def bench_gvn_elim_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gvn_elim", bench_gvn_elim(seed=_SEED + 1100)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gvn_elim bench failed: {exc}") from exc


def bench_instr_sched_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("instr_sched", bench_instr_sched(seed=_SEED + 1101)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"instr_sched bench failed: {exc}") from exc


def bench_licm_hoist_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("licm_hoist", bench_licm_hoist(seed=_SEED + 1102)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"licm_hoist bench failed: {exc}") from exc


def bench_reg_coalesce_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("reg_coalesce", bench_reg_coalesce(seed=_SEED + 1103)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"reg_coalesce bench failed: {exc}") from exc


def bench_sccp_const_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sccp_const", bench_sccp_const(seed=_SEED + 1104)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sccp_const bench failed: {exc}") from exc


def bench_ssa_construct_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ssa_construct", bench_ssa_construct(seed=_SEED + 1105)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ssa_construct bench failed: {exc}") from exc
