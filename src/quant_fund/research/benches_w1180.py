"""Wave-1180 allied-health canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.allied_health import bench_allied_health
from quant_fund.models.midwifery import bench_midwifery
from quant_fund.models.nursing_studies import bench_nursing_studies
from quant_fund.models.occupational_science import bench_occupational_science
from quant_fund.models.paramedicine import bench_paramedicine
from quant_fund.models.speech_pathology import bench_speech_pathology

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


def bench_nursing_studies_family(seed: int = _SEED + 56800) -> dict[str, float]:
    return _finite_blob(bench_nursing_studies(seed))


def bench_allied_health_family(seed: int = _SEED + 56801) -> dict[str, float]:
    return _finite_blob(bench_allied_health(seed))


def bench_midwifery_family(seed: int = _SEED + 56802) -> dict[str, float]:
    return _finite_blob(bench_midwifery(seed))


def bench_paramedicine_family(seed: int = _SEED + 56803) -> dict[str, float]:
    return _finite_blob(bench_paramedicine(seed))


def bench_occupational_science_family(seed: int = _SEED + 56804) -> dict[str, float]:
    return _finite_blob(bench_occupational_science(seed))


def bench_speech_pathology_family(seed: int = _SEED + 56805) -> dict[str, float]:
    return _finite_blob(bench_speech_pathology(seed))
