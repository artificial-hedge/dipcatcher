"""Wave-1192 integrative-medicine canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.acupuncture_studies import bench_acupuncture_studies
from quant_fund.models.chiropractic_studies import bench_chiropractic_studies
from quant_fund.models.herbal_medicine import bench_herbal_medicine
from quant_fund.models.homeopathy import bench_homeopathy
from quant_fund.models.naturopathy import bench_naturopathy
from quant_fund.models.osteopathy_studies import bench_osteopathy_studies

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


def bench_acupuncture_studies_family(seed: int = _SEED + 58000) -> dict[str, float]:
    return _finite_blob(bench_acupuncture_studies(seed))


def bench_chiropractic_studies_family(seed: int = _SEED + 58001) -> dict[str, float]:
    return _finite_blob(bench_chiropractic_studies(seed))


def bench_naturopathy_family(seed: int = _SEED + 58002) -> dict[str, float]:
    return _finite_blob(bench_naturopathy(seed))


def bench_homeopathy_family(seed: int = _SEED + 58003) -> dict[str, float]:
    return _finite_blob(bench_homeopathy(seed))


def bench_herbal_medicine_family(seed: int = _SEED + 58004) -> dict[str, float]:
    return _finite_blob(bench_herbal_medicine(seed))


def bench_osteopathy_studies_family(seed: int = _SEED + 58005) -> dict[str, float]:
    return _finite_blob(bench_osteopathy_studies(seed))
