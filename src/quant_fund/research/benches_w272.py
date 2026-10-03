"""Wave-272 control-theory-2 benches: nonlinear + adaptive control."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.backstepping import bench_backstepping
from quant_fund.models.gain_schedule import bench_gain_schedule
from quant_fund.models.pid_antiwindup import bench_pid_antiwindup
from quant_fund.models.repetitive_ctrl import bench_repetitive_ctrl
from quant_fund.models.sliding_mode import bench_sliding_mode
from quant_fund.models.smith_predictor import bench_smith_predictor

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


def bench_pid_antiwindup_family(seed: int = _SEED + 1490) -> dict[str, float]:
    return _floats(_finite_blob("pid_antiwindup", bench_pid_antiwindup(seed)))


def bench_sliding_mode_family(seed: int = _SEED + 1491) -> dict[str, float]:
    return _floats(_finite_blob("sliding_mode", bench_sliding_mode(seed)))


def bench_gain_schedule_family(seed: int = _SEED + 1492) -> dict[str, float]:
    return _floats(_finite_blob("gain_schedule", bench_gain_schedule(seed)))


def bench_smith_predictor_family(seed: int = _SEED + 1493) -> dict[str, float]:
    return _floats(_finite_blob("smith_predictor", bench_smith_predictor(seed)))


def bench_backstepping_family(seed: int = _SEED + 1494) -> dict[str, float]:
    return _floats(_finite_blob("backstepping", bench_backstepping(seed)))


def bench_repetitive_ctrl_family(seed: int = _SEED + 1495) -> dict[str, float]:
    return _floats(_finite_blob("repetitive_ctrl", bench_repetitive_ctrl(seed)))
