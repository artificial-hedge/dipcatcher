"""Wave-1133 physics-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.conformal_field_theory import bench_conformal_field_theory
from quant_fund.models.holography_ads import bench_holography_ads
from quant_fund.models.lattice_field_theory import bench_lattice_field_theory
from quant_fund.models.loop_quantum_gravity import bench_loop_quantum_gravity
from quant_fund.models.statistical_field_theory import bench_statistical_field_theory
from quant_fund.models.string_theory_math import bench_string_theory_math

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


def bench_statistical_field_theory_family(seed: int = _SEED + 52100) -> dict[str, float]:
    return _finite_blob(bench_statistical_field_theory(seed))


def bench_conformal_field_theory_family(seed: int = _SEED + 52101) -> dict[str, float]:
    return _finite_blob(bench_conformal_field_theory(seed))


def bench_lattice_field_theory_family(seed: int = _SEED + 52102) -> dict[str, float]:
    return _finite_blob(bench_lattice_field_theory(seed))


def bench_string_theory_math_family(seed: int = _SEED + 52103) -> dict[str, float]:
    return _finite_blob(bench_string_theory_math(seed))


def bench_loop_quantum_gravity_family(seed: int = _SEED + 52104) -> dict[str, float]:
    return _finite_blob(bench_loop_quantum_gravity(seed))


def bench_holography_ads_family(seed: int = _SEED + 52105) -> dict[str, float]:
    return _finite_blob(bench_holography_ads(seed))
