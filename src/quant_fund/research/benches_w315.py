"""Wave-315 robotics-5 canon bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dmp_control import bench_dmp_control
from quant_fund.models.ds_motion import bench_ds_motion
from quant_fund.models.grasp_epsilon import bench_grasp_epsilon
from quant_fund.models.rmpflow import bench_rmpflow
from quant_fund.models.rrt_connect import bench_rrt_connect
from quant_fund.models.wbc_qp import bench_wbc_qp

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


def bench_rmpflow_family(seed: int = _SEED + 1797) -> dict[str, float]:
    return _floats(_finite_blob("rmpflow", bench_rmpflow(seed)))


def bench_ds_motion_family(seed: int = _SEED + 1798) -> dict[str, float]:
    return _floats(_finite_blob("ds_motion", bench_ds_motion(seed)))


def bench_wbc_qp_family(seed: int = _SEED + 1799) -> dict[str, float]:
    return _floats(_finite_blob("wbc_qp", bench_wbc_qp(seed)))


def bench_grasp_epsilon_family(seed: int = _SEED + 1800) -> dict[str, float]:
    return _floats(_finite_blob("grasp_epsilon", bench_grasp_epsilon(seed)))


def bench_rrt_connect_family(seed: int = _SEED + 1801) -> dict[str, float]:
    return _floats(_finite_blob("rrt_connect", bench_rrt_connect(seed)))


def bench_dmp_control_family(seed: int = _SEED + 1802) -> dict[str, float]:
    return _floats(_finite_blob("dmp_control", bench_dmp_control(seed)))
