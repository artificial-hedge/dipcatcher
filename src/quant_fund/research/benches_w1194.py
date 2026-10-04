"""Wave-1194 clinical-specialties canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.genetic_counseling import bench_genetic_counseling
from quant_fund.models.lactation_consulting import bench_lactation_consulting
from quant_fund.models.perfusion_technology import bench_perfusion_technology
from quant_fund.models.podiatric_medicine import bench_podiatric_medicine
from quant_fund.models.radiation_therapy import bench_radiation_therapy
from quant_fund.models.respiratory_therapy import bench_respiratory_therapy

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


def bench_genetic_counseling_family(seed: int = _SEED + 58200) -> dict[str, float]:
    return _finite_blob(bench_genetic_counseling(seed))


def bench_lactation_consulting_family(seed: int = _SEED + 58201) -> dict[str, float]:
    return _finite_blob(bench_lactation_consulting(seed))


def bench_podiatric_medicine_family(seed: int = _SEED + 58202) -> dict[str, float]:
    return _finite_blob(bench_podiatric_medicine(seed))


def bench_respiratory_therapy_family(seed: int = _SEED + 58203) -> dict[str, float]:
    return _finite_blob(bench_respiratory_therapy(seed))


def bench_perfusion_technology_family(seed: int = _SEED + 58204) -> dict[str, float]:
    return _finite_blob(bench_perfusion_technology(seed))


def bench_radiation_therapy_family(seed: int = _SEED + 58205) -> dict[str, float]:
    return _finite_blob(bench_radiation_therapy(seed))
