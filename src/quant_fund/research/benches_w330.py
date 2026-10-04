"""Wave-330 SMT-theory canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.array_theory import bench_array_theory
from quant_fund.models.bv_ops import bench_bv_ops
from quant_fund.models.diff_logic import bench_diff_logic
from quant_fund.models.dpllt import bench_dpllt
from quant_fund.models.lia_branch import bench_lia_branch
from quant_fund.models.mcsat_lite import bench_mcsat_lite

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


def bench_diff_logic_family(seed: int = _SEED + 1887) -> dict[str, float]:
    return _floats(_finite_blob("diff_logic", bench_diff_logic(seed)))


def bench_array_theory_family(seed: int = _SEED + 1888) -> dict[str, float]:
    return _floats(_finite_blob("array_theory", bench_array_theory(seed)))


def bench_bv_ops_family(seed: int = _SEED + 1889) -> dict[str, float]:
    return _floats(_finite_blob("bv_ops", bench_bv_ops(seed)))


def bench_dpllt_family(seed: int = _SEED + 1890) -> dict[str, float]:
    return _floats(_finite_blob("dpllt", bench_dpllt(seed)))


def bench_lia_branch_family(seed: int = _SEED + 1891) -> dict[str, float]:
    return _floats(_finite_blob("lia_branch", bench_lia_branch(seed)))


def bench_mcsat_lite_family(seed: int = _SEED + 1892) -> dict[str, float]:
    return _floats(_finite_blob("mcsat_lite", bench_mcsat_lite(seed)))
