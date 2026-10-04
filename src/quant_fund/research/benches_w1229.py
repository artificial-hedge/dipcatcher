"""Wave-1229 pediatrics canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.adolescent_medicine_studies import bench_adolescent_medicine_studies
from quant_fund.models.developmental_pediatrics import bench_developmental_pediatrics
from quant_fund.models.neonatal_medicine_studies import bench_neonatal_medicine_studies
from quant_fund.models.pediatric_cardiology import bench_pediatric_cardiology
from quant_fund.models.pediatric_oncology import bench_pediatric_oncology
from quant_fund.models.pediatrics_studies import bench_pediatrics_studies

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


def bench_pediatrics_studies_family(seed: int = _SEED + 61700) -> dict[str, float]:
    return _finite_blob(bench_pediatrics_studies(seed))


def bench_neonatal_medicine_studies_family(seed: int = _SEED + 61701) -> dict[str, float]:
    return _finite_blob(bench_neonatal_medicine_studies(seed))


def bench_pediatric_cardiology_family(seed: int = _SEED + 61702) -> dict[str, float]:
    return _finite_blob(bench_pediatric_cardiology(seed))


def bench_pediatric_oncology_family(seed: int = _SEED + 61703) -> dict[str, float]:
    return _finite_blob(bench_pediatric_oncology(seed))


def bench_adolescent_medicine_studies_family(seed: int = _SEED + 61704) -> dict[str, float]:
    return _finite_blob(bench_adolescent_medicine_studies(seed))


def bench_developmental_pediatrics_family(seed: int = _SEED + 61705) -> dict[str, float]:
    return _finite_blob(bench_developmental_pediatrics(seed))
