"""Wave-998 singularity/blow-up canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.fujita_exponent import bench_fujita_exponent
from quant_fund.models.matched_asymptotic_pde import bench_matched_asymptotic_pde
from quant_fund.models.regularity_critical import bench_regularity_critical
from quant_fund.models.self_similar_blowup import bench_self_similar_blowup
from quant_fund.models.semilinear_heat import bench_semilinear_heat
from quant_fund.models.singularity_formation import bench_singularity_formation

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


def bench_semilinear_heat_family(seed: int = _SEED + 38600) -> dict[str, float]:
    return _finite_blob(bench_semilinear_heat(seed))


def bench_fujita_exponent_family(seed: int = _SEED + 38601) -> dict[str, float]:
    return _finite_blob(bench_fujita_exponent(seed))


def bench_singularity_formation_family(seed: int = _SEED + 38602) -> dict[str, float]:
    return _finite_blob(bench_singularity_formation(seed))


def bench_matched_asymptotic_pde_family(seed: int = _SEED + 38603) -> dict[str, float]:
    return _finite_blob(bench_matched_asymptotic_pde(seed))


def bench_self_similar_blowup_family(seed: int = _SEED + 38604) -> dict[str, float]:
    return _finite_blob(bench_self_similar_blowup(seed))


def bench_regularity_critical_family(seed: int = _SEED + 38605) -> dict[str, float]:
    return _finite_blob(bench_regularity_critical(seed))
