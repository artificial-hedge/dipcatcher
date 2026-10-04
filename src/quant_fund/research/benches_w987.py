"""Wave-987 parabolic/Li-Yau canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.davies_gaffney import bench_davies_gaffney
from quant_fund.models.gaussian_upper import bench_gaussian_upper
from quant_fund.models.grad_est import bench_grad_est
from quant_fund.models.li_yau import bench_li_yau
from quant_fund.models.nash_ineq import bench_nash_ineq
from quant_fund.models.parabolic_harnack import bench_parabolic_harnack

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in blob.items():
        if key.lower() in _FORBIDDEN:
            raise ValueError(f"forbidden metric key: {key}")
        if not math.isfinite(val):
            raise ValueError(f"non-finite metric: {key}")
        out[key] = float(val)
    return out


def _floats(xs: Iterable[float]) -> list[float]:
    return [float(x) for x in xs]


def bench_parabolic_harnack_family(seed: int = _SEED + 37500) -> dict[str, float]:
    return _finite_blob(bench_parabolic_harnack(seed))


def bench_gaussian_upper_family(seed: int = _SEED + 37501) -> dict[str, float]:
    return _finite_blob(bench_gaussian_upper(seed))


def bench_li_yau_family(seed: int = _SEED + 37502) -> dict[str, float]:
    return _finite_blob(bench_li_yau(seed))


def bench_nash_ineq_family(seed: int = _SEED + 37503) -> dict[str, float]:
    return _finite_blob(bench_nash_ineq(seed))


def bench_davies_gaffney_family(seed: int = _SEED + 37504) -> dict[str, float]:
    return _finite_blob(bench_davies_gaffney(seed))


def bench_grad_est_family(seed: int = _SEED + 37505) -> dict[str, float]:
    return _finite_blob(bench_grad_est(seed))
