"""Wave-1228 dentistry canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.dental_studies import bench_dental_studies
from quant_fund.models.endodontic_studies import bench_endodontic_studies
from quant_fund.models.oral_surgery_studies import bench_oral_surgery_studies
from quant_fund.models.orthodontic_studies import bench_orthodontic_studies
from quant_fund.models.pediatric_dentistry import bench_pediatric_dentistry
from quant_fund.models.periodontal_studies import bench_periodontal_studies

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


def bench_dental_studies_family(seed: int = _SEED + 61600) -> dict[str, float]:
    return _finite_blob(bench_dental_studies(seed))


def bench_oral_surgery_studies_family(seed: int = _SEED + 61601) -> dict[str, float]:
    return _finite_blob(bench_oral_surgery_studies(seed))


def bench_endodontic_studies_family(seed: int = _SEED + 61602) -> dict[str, float]:
    return _finite_blob(bench_endodontic_studies(seed))


def bench_periodontal_studies_family(seed: int = _SEED + 61603) -> dict[str, float]:
    return _finite_blob(bench_periodontal_studies(seed))


def bench_orthodontic_studies_family(seed: int = _SEED + 61604) -> dict[str, float]:
    return _finite_blob(bench_orthodontic_studies(seed))


def bench_pediatric_dentistry_family(seed: int = _SEED + 61605) -> dict[str, float]:
    return _finite_blob(bench_pediatric_dentistry(seed))
