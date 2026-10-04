"""Wave-1130 anthropology-5 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.anthropology_of_religion import bench_anthropology_of_religion
from quant_fund.models.cognitive_anthropology import bench_cognitive_anthropology
from quant_fund.models.kinship_studies import bench_kinship_studies
from quant_fund.models.material_culture import bench_material_culture
from quant_fund.models.museum_anthropology import bench_museum_anthropology
from quant_fund.models.social_anthropology import bench_social_anthropology

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


def bench_social_anthropology_family(seed: int = _SEED + 51800) -> dict[str, float]:
    return _finite_blob(bench_social_anthropology(seed))


def bench_cognitive_anthropology_family(seed: int = _SEED + 51801) -> dict[str, float]:
    return _finite_blob(bench_cognitive_anthropology(seed))


def bench_anthropology_of_religion_family(seed: int = _SEED + 51802) -> dict[str, float]:
    return _finite_blob(bench_anthropology_of_religion(seed))


def bench_kinship_studies_family(seed: int = _SEED + 51803) -> dict[str, float]:
    return _finite_blob(bench_kinship_studies(seed))


def bench_material_culture_family(seed: int = _SEED + 51804) -> dict[str, float]:
    return _finite_blob(bench_material_culture(seed))


def bench_museum_anthropology_family(seed: int = _SEED + 51805) -> dict[str, float]:
    return _finite_blob(bench_museum_anthropology(seed))
