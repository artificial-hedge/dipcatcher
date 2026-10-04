"""Wave-1136 law-4 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.environmental_law import bench_environmental_law
from quant_fund.models.evidence_law import bench_evidence_law
from quant_fund.models.family_law import bench_family_law
from quant_fund.models.immigration_law import bench_immigration_law
from quant_fund.models.labor_law import bench_labor_law
from quant_fund.models.tax_law import bench_tax_law

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


def bench_environmental_law_family(seed: int = _SEED + 52400) -> dict[str, float]:
    return _finite_blob(bench_environmental_law(seed))


def bench_family_law_family(seed: int = _SEED + 52401) -> dict[str, float]:
    return _finite_blob(bench_family_law(seed))


def bench_labor_law_family(seed: int = _SEED + 52402) -> dict[str, float]:
    return _finite_blob(bench_labor_law(seed))


def bench_tax_law_family(seed: int = _SEED + 52403) -> dict[str, float]:
    return _finite_blob(bench_tax_law(seed))


def bench_evidence_law_family(seed: int = _SEED + 52404) -> dict[str, float]:
    return _finite_blob(bench_evidence_law(seed))


def bench_immigration_law_family(seed: int = _SEED + 52405) -> dict[str, float]:
    return _finite_blob(bench_immigration_law(seed))
