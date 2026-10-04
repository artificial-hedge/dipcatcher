"""Wave-1050 pharmacology canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.clinical_pharmacology import bench_clinical_pharmacology
from quant_fund.models.drug_metabolism import bench_drug_metabolism
from quant_fund.models.neuropharmacology import bench_neuropharmacology
from quant_fund.models.pharmacodynamics import bench_pharmacodynamics
from quant_fund.models.pharmacokinetics_2 import bench_pharmacokinetics_2
from quant_fund.models.toxicology import bench_toxicology

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


def bench_pharmacodynamics_family(seed: int = _SEED + 43800) -> dict[str, float]:
    return _finite_blob(bench_pharmacodynamics(seed))


def bench_pharmacokinetics_2_family(seed: int = _SEED + 43801) -> dict[str, float]:
    return _finite_blob(bench_pharmacokinetics_2(seed))


def bench_toxicology_family(seed: int = _SEED + 43802) -> dict[str, float]:
    return _finite_blob(bench_toxicology(seed))


def bench_clinical_pharmacology_family(seed: int = _SEED + 43803) -> dict[str, float]:
    return _finite_blob(bench_clinical_pharmacology(seed))


def bench_neuropharmacology_family(seed: int = _SEED + 43804) -> dict[str, float]:
    return _finite_blob(bench_neuropharmacology(seed))


def bench_drug_metabolism_family(seed: int = _SEED + 43805) -> dict[str, float]:
    return _finite_blob(bench_drug_metabolism(seed))
