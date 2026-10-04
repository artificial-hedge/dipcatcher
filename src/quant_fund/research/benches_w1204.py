"""Wave-1204 geriatric-care canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.aging_research import bench_aging_research
from quant_fund.models.geriatric_medicine import bench_geriatric_medicine
from quant_fund.models.gerontology_studies import bench_gerontology_studies
from quant_fund.models.hospice_care import bench_hospice_care
from quant_fund.models.longevity_medicine import bench_longevity_medicine
from quant_fund.models.palliative_care import bench_palliative_care

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


def bench_geriatric_medicine_family(seed: int = _SEED + 59200) -> dict[str, float]:
    return _finite_blob(bench_geriatric_medicine(seed))


def bench_palliative_care_family(seed: int = _SEED + 59201) -> dict[str, float]:
    return _finite_blob(bench_palliative_care(seed))


def bench_hospice_care_family(seed: int = _SEED + 59202) -> dict[str, float]:
    return _finite_blob(bench_hospice_care(seed))


def bench_gerontology_studies_family(seed: int = _SEED + 59203) -> dict[str, float]:
    return _finite_blob(bench_gerontology_studies(seed))


def bench_aging_research_family(seed: int = _SEED + 59204) -> dict[str, float]:
    return _finite_blob(bench_aging_research(seed))


def bench_longevity_medicine_family(seed: int = _SEED + 59205) -> dict[str, float]:
    return _finite_blob(bench_longevity_medicine(seed))
