"""Wave-1066 area studies canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.african_studies import bench_african_studies
from quant_fund.models.asian_studies import bench_asian_studies
from quant_fund.models.european_studies import bench_european_studies
from quant_fund.models.latin_american_studies import bench_latin_american_studies
from quant_fund.models.middle_eastern_studies import bench_middle_eastern_studies
from quant_fund.models.slavic_studies import bench_slavic_studies

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


def bench_latin_american_studies_family(seed: int = _SEED + 45400) -> dict[str, float]:
    return _finite_blob(bench_latin_american_studies(seed))


def bench_asian_studies_family(seed: int = _SEED + 45401) -> dict[str, float]:
    return _finite_blob(bench_asian_studies(seed))


def bench_european_studies_family(seed: int = _SEED + 45402) -> dict[str, float]:
    return _finite_blob(bench_european_studies(seed))


def bench_middle_eastern_studies_family(seed: int = _SEED + 45403) -> dict[str, float]:
    return _finite_blob(bench_middle_eastern_studies(seed))


def bench_african_studies_family(seed: int = _SEED + 45404) -> dict[str, float]:
    return _finite_blob(bench_african_studies(seed))


def bench_slavic_studies_family(seed: int = _SEED + 45405) -> dict[str, float]:
    return _finite_blob(bench_slavic_studies(seed))
