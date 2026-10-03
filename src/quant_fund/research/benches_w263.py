"""Wave-263 real-time canon adapter: SYNTHETIC benches."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.debounce_fsm import bench_debounce_fsm
from quant_fund.models.edf_scheduler import bench_edf_scheduler
from quant_fund.models.ring_buffer import bench_ring_buffer
from quant_fund.models.rms_scheduler import bench_rms_scheduler
from quant_fund.models.watchdog_task import bench_watchdog_task
from quant_fund.models.wcet_est import bench_wcet_est

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


def bench_edf_scheduler_family(seed: int = _SEED + 1400) -> dict[str, float]:
    return bench_edf_scheduler(seed)


def bench_rms_scheduler_family(seed: int = _SEED + 1401) -> dict[str, float]:
    return bench_rms_scheduler(seed)


def bench_wcet_est_family(seed: int = _SEED + 1402) -> dict[str, float]:
    return bench_wcet_est(seed)


def bench_debounce_fsm_family(seed: int = _SEED + 1403) -> dict[str, float]:
    return bench_debounce_fsm(seed)


def bench_watchdog_task_family(seed: int = _SEED + 1404) -> dict[str, float]:
    return bench_watchdog_task(seed)


def bench_ring_buffer_family(seed: int = _SEED + 1405) -> dict[str, float]:
    return bench_ring_buffer(seed)
