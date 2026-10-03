"""Wave-219 adapters: SAT/symbolic-reasoning canon — cdcl_solver,
walksat, unit_propagation, twosat_scc, bdd_ops, ltl_mc —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bdd_ops import bench_bdd_ops
from quant_fund.models.cdcl_solver import bench_cdcl_solver
from quant_fund.models.ltl_mc import bench_ltl_mc
from quant_fund.models.twosat_scc import bench_twosat_scc
from quant_fund.models.unit_propagation import bench_unit_propagation
from quant_fund.models.walksat import bench_walksat

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


def bench_bdd_ops_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bdd_ops", bench_bdd_ops(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bdd_ops bench failed: {exc}") from exc


def bench_cdcl_solver_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cdcl_solver", bench_cdcl_solver(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cdcl_solver bench failed: {exc}") from exc


def bench_twosat_scc_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("twosat_scc", bench_twosat_scc(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"twosat_scc bench failed: {exc}") from exc


def bench_walksat_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("walksat", bench_walksat(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"walksat bench failed: {exc}") from exc


def bench_unit_propagation_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("unit_propagation", bench_unit_propagation(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"unit_propagation bench failed: {exc}") from exc


def bench_ltl_mc_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ltl_mc", bench_ltl_mc(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ltl_mc bench failed: {exc}") from exc
