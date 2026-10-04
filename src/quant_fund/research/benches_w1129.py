"""Wave-1129 philosophy-5 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.african_philosophy import bench_african_philosophy
from quant_fund.models.bioethics import bench_bioethics
from quant_fund.models.environmental_philosophy import bench_environmental_philosophy
from quant_fund.models.feminist_philosophy import bench_feminist_philosophy
from quant_fund.models.philosophy_of_education import bench_philosophy_of_education
from quant_fund.models.philosophy_of_medicine import bench_philosophy_of_medicine

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


def bench_bioethics_family(seed: int = _SEED + 51700) -> dict[str, float]:
    return _finite_blob(bench_bioethics(seed))


def bench_philosophy_of_education_family(seed: int = _SEED + 51701) -> dict[str, float]:
    return _finite_blob(bench_philosophy_of_education(seed))


def bench_feminist_philosophy_family(seed: int = _SEED + 51702) -> dict[str, float]:
    return _finite_blob(bench_feminist_philosophy(seed))


def bench_african_philosophy_family(seed: int = _SEED + 51703) -> dict[str, float]:
    return _finite_blob(bench_african_philosophy(seed))


def bench_environmental_philosophy_family(seed: int = _SEED + 51704) -> dict[str, float]:
    return _finite_blob(bench_environmental_philosophy(seed))


def bench_philosophy_of_medicine_family(seed: int = _SEED + 51705) -> dict[str, float]:
    return _finite_blob(bench_philosophy_of_medicine(seed))
