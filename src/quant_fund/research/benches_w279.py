"""Wave-279 control-theory-3 benches: observers, adaptation, feedforward."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dist_obsv import bench_dist_obsv
from quant_fund.models.flat_track import bench_flat_track
from quant_fund.models.l2_gain import bench_l2_gain
from quant_fund.models.luen_obsv import bench_luen_obsv
from quant_fund.models.lyap_synth import bench_lyap_synth
from quant_fund.models.mrac_adapt import bench_mrac_adapt

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


def bench_luen_obsv_family(seed: int = _SEED + 1560) -> dict[str, float]:
    return _floats(_finite_blob("luen_obsv", bench_luen_obsv(seed)))


def bench_dist_obsv_family(seed: int = _SEED + 1561) -> dict[str, float]:
    return _floats(_finite_blob("dist_obsv", bench_dist_obsv(seed)))


def bench_mrac_adapt_family(seed: int = _SEED + 1562) -> dict[str, float]:
    return _floats(_finite_blob("mrac_adapt", bench_mrac_adapt(seed)))


def bench_flat_track_family(seed: int = _SEED + 1563) -> dict[str, float]:
    return _floats(_finite_blob("flat_track", bench_flat_track(seed)))


def bench_lyap_synth_family(seed: int = _SEED + 1564) -> dict[str, float]:
    return _floats(_finite_blob("lyap_synth", bench_lyap_synth(seed)))


def bench_l2_gain_family(seed: int = _SEED + 1565) -> dict[str, float]:
    return _floats(_finite_blob("l2_gain", bench_l2_gain(seed)))
