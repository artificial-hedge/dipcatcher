"""Wave-954 matrix-approximation canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.condition_number import bench_condition_number
from quant_fund.models.low_rank_approx import bench_low_rank_approx
from quant_fund.models.matrix_truncate import bench_matrix_truncate
from quant_fund.models.nuclear_norm import bench_nuclear_norm
from quant_fund.models.rank_estimate import bench_rank_estimate
from quant_fund.models.spectral_threshold import bench_spectral_threshold

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


def bench_low_rank_approx_family(seed: int = _SEED + 34200) -> dict[str, float]:
    return _finite_blob(bench_low_rank_approx(seed))


def bench_nuclear_norm_family(seed: int = _SEED + 34201) -> dict[str, float]:
    return _finite_blob(bench_nuclear_norm(seed))


def bench_spectral_threshold_family(seed: int = _SEED + 34202) -> dict[str, float]:
    return _finite_blob(bench_spectral_threshold(seed))


def bench_matrix_truncate_family(seed: int = _SEED + 34203) -> dict[str, float]:
    return _finite_blob(bench_matrix_truncate(seed))


def bench_rank_estimate_family(seed: int = _SEED + 34204) -> dict[str, float]:
    return _finite_blob(bench_rank_estimate(seed))


def bench_condition_number_family(seed: int = _SEED + 34205) -> dict[str, float]:
    return _finite_blob(bench_condition_number(seed))
