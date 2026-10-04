"""Wave-566 L-functions/random-matrix bench adapters (SYNTHETIC-only)."""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.gue_statistics import bench_gue_statistics
from quant_fund.models.keating_snaith import bench_keating_snaith
from quant_fund.models.montgomery_pair import (
    bench_montgomery_pair,
)
from quant_fund.models.rudnick_sarnak import bench_rudnick_sarnak
from quant_fund.models.selberg_trace2 import bench_selberg_trace2
from quant_fund.models.zero_spacing import bench_zero_spacing

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


def bench_selberg_trace2_family(
    seed: int = _SEED + 3302,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "selberg_trace2",
            bench_selberg_trace2(seed),
        )
    )


def bench_zero_spacing_family(seed: int = _SEED + 3303) -> dict[str, float]:
    return _floats(_finite_blob("zero_spacing", bench_zero_spacing(seed)))


def bench_montgomery_pair_family(
    seed: int = _SEED + 3304,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "montgomery_pair",
            bench_montgomery_pair(seed),
        )
    )


def bench_gue_statistics_family(
    seed: int = _SEED + 3305,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "gue_statistics",
            bench_gue_statistics(seed),
        )
    )


def bench_keating_snaith_family(
    seed: int = _SEED + 3306,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "keating_snaith",
            bench_keating_snaith(seed),
        )
    )


def bench_rudnick_sarnak_family(
    seed: int = _SEED + 3307,
) -> dict[str, float]:
    return _floats(
        _finite_blob(
            "rudnick_sarnak",
            bench_rudnick_sarnak(seed),
        )
    )
