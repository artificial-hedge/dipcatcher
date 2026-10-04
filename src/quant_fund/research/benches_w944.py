"""Wave-944 matrix-analysis-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cauchy_binet import bench_cauchy_binet
from quant_fund.models.fan_inequality import bench_fan_inequality
from quant_fund.models.horn_inequality import bench_horn_inequality
from quant_fund.models.majorization_vec import bench_majorization_vec
from quant_fund.models.schur_complement import bench_schur_complement
from quant_fund.models.weyl_ineq import bench_weyl_ineq

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


def bench_fan_inequality_family(seed: int = _SEED + 33200) -> dict[str, float]:
    return _finite_blob(bench_fan_inequality(seed))


def bench_horn_inequality_family(seed: int = _SEED + 33201) -> dict[str, float]:
    return _finite_blob(bench_horn_inequality(seed))


def bench_weyl_ineq_family(seed: int = _SEED + 33202) -> dict[str, float]:
    return _finite_blob(bench_weyl_ineq(seed))


def bench_cauchy_binet_family(seed: int = _SEED + 33203) -> dict[str, float]:
    return _finite_blob(bench_cauchy_binet(seed))


def bench_schur_complement_family(seed: int = _SEED + 33204) -> dict[str, float]:
    return _finite_blob(bench_schur_complement(seed))


def bench_majorization_vec_family(seed: int = _SEED + 33205) -> dict[str, float]:
    return _finite_blob(bench_majorization_vec(seed))
