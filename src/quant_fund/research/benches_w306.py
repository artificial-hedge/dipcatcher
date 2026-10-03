"""Wave-306 astronomy-3/IOD canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.batch_od import bench_batch_od
from quant_fund.models.cowell_j2 import bench_cowell_j2
from quant_fund.models.cr3bp_dynamics import bench_cr3bp_dynamics
from quant_fund.models.davenport_q import bench_davenport_q
from quant_fund.models.laplace_iod import bench_laplace_iod
from quant_fund.models.porkchop_grid import bench_porkchop_grid

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


def bench_laplace_iod_family(seed: int = _SEED + 1742) -> dict[str, float]:
    return _floats(_finite_blob("laplace_iod", bench_laplace_iod(seed)))


def bench_cowell_j2_family(seed: int = _SEED + 1743) -> dict[str, float]:
    return _floats(_finite_blob("cowell_j2", bench_cowell_j2(seed)))


def bench_batch_od_family(seed: int = _SEED + 1744) -> dict[str, float]:
    return _floats(_finite_blob("batch_od", bench_batch_od(seed)))


def bench_cr3bp_dynamics_family(seed: int = _SEED + 1745) -> dict[str, float]:
    return _floats(_finite_blob("cr3bp_dynamics", bench_cr3bp_dynamics(seed)))


def bench_porkchop_grid_family(seed: int = _SEED + 1746) -> dict[str, float]:
    return _floats(_finite_blob("porkchop_grid", bench_porkchop_grid(seed)))


def bench_davenport_q_family(seed: int = _SEED + 1747) -> dict[str, float]:
    return _floats(_finite_blob("davenport_q", bench_davenport_q(seed)))
