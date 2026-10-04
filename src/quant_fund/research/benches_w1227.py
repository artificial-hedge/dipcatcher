"""Wave-1227 radiology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.body_imaging import bench_body_imaging
from quant_fund.models.diagnostic_imaging import bench_diagnostic_imaging
from quant_fund.models.interventional_neuroradiology import bench_interventional_neuroradiology
from quant_fund.models.musculoskeletal_imaging import bench_musculoskeletal_imaging
from quant_fund.models.pediatric_imaging import bench_pediatric_imaging
from quant_fund.models.radiology_studies import bench_radiology_studies

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


def bench_radiology_studies_family(seed: int = _SEED + 61500) -> dict[str, float]:
    return _finite_blob(bench_radiology_studies(seed))


def bench_diagnostic_imaging_family(seed: int = _SEED + 61501) -> dict[str, float]:
    return _finite_blob(bench_diagnostic_imaging(seed))


def bench_interventional_neuroradiology_family(seed: int = _SEED + 61502) -> dict[str, float]:
    return _finite_blob(bench_interventional_neuroradiology(seed))


def bench_pediatric_imaging_family(seed: int = _SEED + 61503) -> dict[str, float]:
    return _finite_blob(bench_pediatric_imaging(seed))


def bench_musculoskeletal_imaging_family(seed: int = _SEED + 61504) -> dict[str, float]:
    return _finite_blob(bench_musculoskeletal_imaging(seed))


def bench_body_imaging_family(seed: int = _SEED + 61505) -> dict[str, float]:
    return _finite_blob(bench_body_imaging(seed))
