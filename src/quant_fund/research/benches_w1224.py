"""Wave-1224 obgyn canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.fetal_medicine import bench_fetal_medicine
from quant_fund.models.gynecologic_oncology import bench_gynecologic_oncology
from quant_fund.models.gynecology_studies import bench_gynecology_studies
from quant_fund.models.maternal_fetal_medicine import bench_maternal_fetal_medicine
from quant_fund.models.obstetrics_studies import bench_obstetrics_studies
from quant_fund.models.reproductive_endocrinology import bench_reproductive_endocrinology

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


def bench_obstetrics_studies_family(seed: int = _SEED + 61200) -> dict[str, float]:
    return _finite_blob(bench_obstetrics_studies(seed))


def bench_gynecology_studies_family(seed: int = _SEED + 61201) -> dict[str, float]:
    return _finite_blob(bench_gynecology_studies(seed))


def bench_maternal_fetal_medicine_family(seed: int = _SEED + 61202) -> dict[str, float]:
    return _finite_blob(bench_maternal_fetal_medicine(seed))


def bench_reproductive_endocrinology_family(seed: int = _SEED + 61203) -> dict[str, float]:
    return _finite_blob(bench_reproductive_endocrinology(seed))


def bench_gynecologic_oncology_family(seed: int = _SEED + 61204) -> dict[str, float]:
    return _finite_blob(bench_gynecologic_oncology(seed))


def bench_fetal_medicine_family(seed: int = _SEED + 61205) -> dict[str, float]:
    return _finite_blob(bench_fetal_medicine(seed))
