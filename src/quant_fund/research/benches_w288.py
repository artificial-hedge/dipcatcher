"""Wave-288 measure-theory canon bench adapters (deterministic, SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.conv_prob import bench_conv_prob
from quant_fund.models.fubini_swap import bench_fubini_swap
from quant_fund.models.leb_integral import bench_leb_integral
from quant_fund.models.leb_measure import bench_leb_measure
from quant_fund.models.radon_nikodym import bench_radon_nikodym
from quant_fund.models.weak_conv import bench_weak_conv

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


def bench_leb_measure_family(seed: int = _SEED + 1634) -> dict[str, float]:
    return _floats(_finite_blob("leb_measure", bench_leb_measure(seed)))


def bench_leb_integral_family(seed: int = _SEED + 1635) -> dict[str, float]:
    return _floats(_finite_blob("leb_integral", bench_leb_integral(seed)))


def bench_conv_prob_family(seed: int = _SEED + 1636) -> dict[str, float]:
    return _floats(_finite_blob("conv_prob", bench_conv_prob(seed)))


def bench_weak_conv_family(seed: int = _SEED + 1637) -> dict[str, float]:
    return _floats(_finite_blob("weak_conv", bench_weak_conv(seed)))


def bench_fubini_swap_family(seed: int = _SEED + 1638) -> dict[str, float]:
    return _floats(_finite_blob("fubini_swap", bench_fubini_swap(seed)))


def bench_radon_nikodym_family(seed: int = _SEED + 1639) -> dict[str, float]:
    return _floats(_finite_blob("radon_nikodym", bench_radon_nikodym(seed)))
