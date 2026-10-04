"""Wave-1134 computational-math canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.adaptive_method_theory import bench_adaptive_method_theory
from quant_fund.models.finite_element_theory import bench_finite_element_theory
from quant_fund.models.high_performance_numerics import bench_high_performance_numerics
from quant_fund.models.reduced_order_modeling import bench_reduced_order_modeling
from quant_fund.models.spectral_theory_numerics import bench_spectral_theory_numerics
from quant_fund.models.uncertainty_quantification_2 import bench_uncertainty_quantification_2

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


def bench_finite_element_theory_family(seed: int = _SEED + 52200) -> dict[str, float]:
    return _finite_blob(bench_finite_element_theory(seed))


def bench_spectral_theory_numerics_family(seed: int = _SEED + 52201) -> dict[str, float]:
    return _finite_blob(bench_spectral_theory_numerics(seed))


def bench_adaptive_method_theory_family(seed: int = _SEED + 52202) -> dict[str, float]:
    return _finite_blob(bench_adaptive_method_theory(seed))


def bench_reduced_order_modeling_family(seed: int = _SEED + 52203) -> dict[str, float]:
    return _finite_blob(bench_reduced_order_modeling(seed))


def bench_uncertainty_quantification_2_family(seed: int = _SEED + 52204) -> dict[str, float]:
    return _finite_blob(bench_uncertainty_quantification_2(seed))


def bench_high_performance_numerics_family(seed: int = _SEED + 52205) -> dict[str, float]:
    return _finite_blob(bench_high_performance_numerics(seed))
