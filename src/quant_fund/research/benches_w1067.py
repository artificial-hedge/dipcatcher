"""Wave-1067 library/information science canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.archival_studies import bench_archival_studies
from quant_fund.models.digital_humanities import bench_digital_humanities
from quant_fund.models.information_science import bench_information_science
from quant_fund.models.knowledge_organization import bench_knowledge_organization
from quant_fund.models.library_science import bench_library_science
from quant_fund.models.museum_studies import bench_museum_studies

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


def bench_library_science_family(seed: int = _SEED + 45500) -> dict[str, float]:
    return _finite_blob(bench_library_science(seed))


def bench_information_science_family(seed: int = _SEED + 45501) -> dict[str, float]:
    return _finite_blob(bench_information_science(seed))


def bench_archival_studies_family(seed: int = _SEED + 45502) -> dict[str, float]:
    return _finite_blob(bench_archival_studies(seed))


def bench_museum_studies_family(seed: int = _SEED + 45503) -> dict[str, float]:
    return _finite_blob(bench_museum_studies(seed))


def bench_digital_humanities_family(seed: int = _SEED + 45504) -> dict[str, float]:
    return _finite_blob(bench_digital_humanities(seed))


def bench_knowledge_organization_family(seed: int = _SEED + 45505) -> dict[str, float]:
    return _finite_blob(bench_knowledge_organization(seed))
