"""Wave-958 matrix-inequalities canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.araki_lieb_thirring import bench_araki_lieb_thirring
from quant_fund.models.hadamard_fischer import bench_hadamard_fischer
from quant_fund.models.ky_fan import bench_ky_fan
from quant_fund.models.lidskii_thm import bench_lidskii_thm
from quant_fund.models.pinching_ineq import bench_pinching_ineq
from quant_fund.models.von_neumann_trace import bench_von_neumann_trace

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


def bench_ky_fan_family(seed: int = _SEED + 34600) -> dict[str, float]:
    return _finite_blob(bench_ky_fan(seed))


def bench_lidskii_thm_family(seed: int = _SEED + 34601) -> dict[str, float]:
    return _finite_blob(bench_lidskii_thm(seed))


def bench_von_neumann_trace_family(seed: int = _SEED + 34602) -> dict[str, float]:
    return _finite_blob(bench_von_neumann_trace(seed))


def bench_pinching_ineq_family(seed: int = _SEED + 34603) -> dict[str, float]:
    return _finite_blob(bench_pinching_ineq(seed))


def bench_araki_lieb_thirring_family(seed: int = _SEED + 34604) -> dict[str, float]:
    return _finite_blob(bench_araki_lieb_thirring(seed))


def bench_hadamard_fischer_family(seed: int = _SEED + 34605) -> dict[str, float]:
    return _finite_blob(bench_hadamard_fischer(seed))
