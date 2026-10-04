"""Wave-1197 counseling-neonatal canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.addiction_counseling import bench_addiction_counseling
from quant_fund.models.genetic_screening import bench_genetic_screening
from quant_fund.models.neonatology_studies import bench_neonatology_studies
from quant_fund.models.pediatric_therapeutics import bench_pediatric_therapeutics
from quant_fund.models.prenatal_studies import bench_prenatal_studies
from quant_fund.models.rehabilitation_counseling import bench_rehabilitation_counseling

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


def bench_addiction_counseling_family(seed: int = _SEED + 58500) -> dict[str, float]:
    return _finite_blob(bench_addiction_counseling(seed))


def bench_rehabilitation_counseling_family(seed: int = _SEED + 58501) -> dict[str, float]:
    return _finite_blob(bench_rehabilitation_counseling(seed))


def bench_genetic_screening_family(seed: int = _SEED + 58502) -> dict[str, float]:
    return _finite_blob(bench_genetic_screening(seed))


def bench_prenatal_studies_family(seed: int = _SEED + 58503) -> dict[str, float]:
    return _finite_blob(bench_prenatal_studies(seed))


def bench_neonatology_studies_family(seed: int = _SEED + 58504) -> dict[str, float]:
    return _finite_blob(bench_neonatology_studies(seed))


def bench_pediatric_therapeutics_family(seed: int = _SEED + 58505) -> dict[str, float]:
    return _finite_blob(bench_pediatric_therapeutics(seed))
