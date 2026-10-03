"""Wave-978 harmonic-analysis-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.bochner_riesz import bench_bochner_riesz
from quant_fund.models.hausdorff_young import bench_hausdorff_young
from quant_fund.models.lp_multiplier import bench_lp_multiplier
from quant_fund.models.oscillatory_int import bench_oscillatory_int
from quant_fund.models.restriction_est import bench_restriction_est
from quant_fund.models.strichartz_est import bench_strichartz_est

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


def bench_hausdorff_young_family(seed: int = _SEED + 36600) -> dict[str, float]:
    return _finite_blob(bench_hausdorff_young(seed))


def bench_restriction_est_family(seed: int = _SEED + 36601) -> dict[str, float]:
    return _finite_blob(bench_restriction_est(seed))


def bench_bochner_riesz_family(seed: int = _SEED + 36602) -> dict[str, float]:
    return _finite_blob(bench_bochner_riesz(seed))


def bench_lp_multiplier_family(seed: int = _SEED + 36603) -> dict[str, float]:
    return _finite_blob(bench_lp_multiplier(seed))


def bench_oscillatory_int_family(seed: int = _SEED + 36604) -> dict[str, float]:
    return _finite_blob(bench_oscillatory_int(seed))


def bench_strichartz_est_family(seed: int = _SEED + 36605) -> dict[str, float]:
    return _finite_blob(bench_strichartz_est(seed))
