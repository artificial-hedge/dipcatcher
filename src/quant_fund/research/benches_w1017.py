"""Wave-1017 continuum-mechanics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.navier_cauchy import bench_navier_cauchy
from quant_fund.models.plasticity import bench_plasticity
from quant_fund.models.poroelasticity import bench_poroelasticity
from quant_fund.models.rheology import bench_rheology
from quant_fund.models.stress_tensor import bench_stress_tensor
from quant_fund.models.viscoelasticity import bench_viscoelasticity

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


def bench_navier_cauchy_family(seed: int = _SEED + 40500) -> dict[str, float]:
    return _finite_blob(bench_navier_cauchy(seed))


def bench_stress_tensor_family(seed: int = _SEED + 40501) -> dict[str, float]:
    return _finite_blob(bench_stress_tensor(seed))


def bench_rheology_family(seed: int = _SEED + 40502) -> dict[str, float]:
    return _finite_blob(bench_rheology(seed))


def bench_viscoelasticity_family(seed: int = _SEED + 40503) -> dict[str, float]:
    return _finite_blob(bench_viscoelasticity(seed))


def bench_plasticity_family(seed: int = _SEED + 40504) -> dict[str, float]:
    return _finite_blob(bench_plasticity(seed))


def bench_poroelasticity_family(seed: int = _SEED + 40505) -> dict[str, float]:
    return _finite_blob(bench_poroelasticity(seed))
