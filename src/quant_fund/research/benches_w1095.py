"""Wave-1095 sociology-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.deviance_studies import bench_deviance_studies
from quant_fund.models.family_sociology import bench_family_sociology
from quant_fund.models.medical_sociology import bench_medical_sociology
from quant_fund.models.organization_theory import bench_organization_theory
from quant_fund.models.rural_sociology import bench_rural_sociology
from quant_fund.models.social_movements import bench_social_movements

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


def bench_medical_sociology_family(seed: int = _SEED + 48300) -> dict[str, float]:
    return _finite_blob(bench_medical_sociology(seed))


def bench_deviance_studies_family(seed: int = _SEED + 48301) -> dict[str, float]:
    return _finite_blob(bench_deviance_studies(seed))


def bench_family_sociology_family(seed: int = _SEED + 48302) -> dict[str, float]:
    return _finite_blob(bench_family_sociology(seed))


def bench_organization_theory_family(seed: int = _SEED + 48303) -> dict[str, float]:
    return _finite_blob(bench_organization_theory(seed))


def bench_social_movements_family(seed: int = _SEED + 48304) -> dict[str, float]:
    return _finite_blob(bench_social_movements(seed))


def bench_rural_sociology_family(seed: int = _SEED + 48305) -> dict[str, float]:
    return _finite_blob(bench_rural_sociology(seed))
