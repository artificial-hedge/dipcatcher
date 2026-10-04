"""Wave-296 robotics-4 canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.chomp import bench_chomp
from quant_fund.models.gjk_epa import bench_gjk_epa
from quant_fund.models.ilqr import bench_ilqr
from quant_fund.models.lqr_funnel import bench_lqr_funnel
from quant_fund.models.rts_smoother import bench_rts_smoother
from quant_fund.models.se3_spline import bench_se3_spline

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


def bench_lqr_funnel_family(seed: int = _SEED + 1682) -> dict[str, float]:
    return _floats(_finite_blob("lqr_funnel", bench_lqr_funnel(seed)))


def bench_chomp_family(seed: int = _SEED + 1683) -> dict[str, float]:
    return _floats(_finite_blob("chomp", bench_chomp(seed)))


def bench_gjk_epa_family(seed: int = _SEED + 1684) -> dict[str, float]:
    return _floats(_finite_blob("gjk_epa", bench_gjk_epa(seed)))


def bench_ilqr_family(seed: int = _SEED + 1685) -> dict[str, float]:
    return _floats(_finite_blob("ilqr", bench_ilqr(seed)))


def bench_rts_smoother_family(seed: int = _SEED + 1686) -> dict[str, float]:
    return _floats(_finite_blob("rts_smoother", bench_rts_smoother(seed)))


def bench_se3_spline_family(seed: int = _SEED + 1687) -> dict[str, float]:
    return _floats(_finite_blob("se3_spline", bench_se3_spline(seed)))
