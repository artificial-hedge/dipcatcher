"""Wave-1092 art-historiography canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.connoisseurship import bench_connoisseurship
from quant_fund.models.curation_practice import bench_curation_practice
from quant_fund.models.formal_analysis import bench_formal_analysis
from quant_fund.models.iconography import bench_iconography
from quant_fund.models.iconology import bench_iconology
from quant_fund.models.provenance_studies import bench_provenance_studies

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


def bench_iconography_family(seed: int = _SEED + 48000) -> dict[str, float]:
    return _finite_blob(bench_iconography(seed))


def bench_iconology_family(seed: int = _SEED + 48001) -> dict[str, float]:
    return _finite_blob(bench_iconology(seed))


def bench_connoisseurship_family(seed: int = _SEED + 48002) -> dict[str, float]:
    return _finite_blob(bench_connoisseurship(seed))


def bench_provenance_studies_family(seed: int = _SEED + 48003) -> dict[str, float]:
    return _finite_blob(bench_provenance_studies(seed))


def bench_curation_practice_family(seed: int = _SEED + 48004) -> dict[str, float]:
    return _finite_blob(bench_curation_practice(seed))


def bench_formal_analysis_family(seed: int = _SEED + 48005) -> dict[str, float]:
    return _finite_blob(bench_formal_analysis(seed))
