"""Wave-1199 interventional-medicine canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.cardiac_electrophysiology import bench_cardiac_electrophysiology
from quant_fund.models.dialysis_technology import bench_dialysis_technology
from quant_fund.models.hepatobiliary_studies import bench_hepatobiliary_studies
from quant_fund.models.interventional_radiology import bench_interventional_radiology
from quant_fund.models.nuclear_cardiology import bench_nuclear_cardiology
from quant_fund.models.transplant_studies import bench_transplant_studies

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


def bench_dialysis_technology_family(seed: int = _SEED + 58700) -> dict[str, float]:
    return _finite_blob(bench_dialysis_technology(seed))


def bench_transplant_studies_family(seed: int = _SEED + 58701) -> dict[str, float]:
    return _finite_blob(bench_transplant_studies(seed))


def bench_hepatobiliary_studies_family(seed: int = _SEED + 58702) -> dict[str, float]:
    return _finite_blob(bench_hepatobiliary_studies(seed))


def bench_cardiac_electrophysiology_family(seed: int = _SEED + 58703) -> dict[str, float]:
    return _finite_blob(bench_cardiac_electrophysiology(seed))


def bench_interventional_radiology_family(seed: int = _SEED + 58704) -> dict[str, float]:
    return _finite_blob(bench_interventional_radiology(seed))


def bench_nuclear_cardiology_family(seed: int = _SEED + 58705) -> dict[str, float]:
    return _finite_blob(bench_nuclear_cardiology(seed))
