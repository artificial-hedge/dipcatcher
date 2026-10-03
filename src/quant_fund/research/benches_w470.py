"""Wave-470 homotopy-9 bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.e_infty2 import bench_e_infty2
from quant_fund.models.h_space import bench_h_space
from quant_fund.models.james_constr import bench_james_constr
from quant_fund.models.obstruction_th import bench_obstruction_th
from quant_fund.models.power_op import bench_power_op
from quant_fund.models.rational_htpy import bench_rational_htpy

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


def bench_e_infty2_family(seed: int = _SEED + 2726) -> dict[str, float]:
    return _floats(_finite_blob("e_infty2", bench_e_infty2(seed)))


def bench_power_op_family(seed: int = _SEED + 2727) -> dict[str, float]:
    return _floats(_finite_blob("power_op", bench_power_op(seed)))


def bench_obstruction_th_family(seed: int = _SEED + 2728) -> dict[str, float]:
    return _floats(_finite_blob("obstruction_th", bench_obstruction_th(seed)))


def bench_rational_htpy_family(seed: int = _SEED + 2729) -> dict[str, float]:
    return _floats(_finite_blob("rational_htpy", bench_rational_htpy(seed)))


def bench_h_space_family(seed: int = _SEED + 2730) -> dict[str, float]:
    return _floats(_finite_blob("h_space", bench_h_space(seed)))


def bench_james_constr_family(seed: int = _SEED + 2731) -> dict[str, float]:
    return _floats(_finite_blob("james_constr", bench_james_constr(seed)))
