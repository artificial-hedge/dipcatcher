"""Wave-1193 allied-health-2 canon benchmark adapters (SYNTHETIC)."""

from __future__ import annotations

import math
from collections.abc import Iterable

from quant_fund.models.audiology_studies import bench_audiology_studies
from quant_fund.models.clinical_psychology_2 import bench_clinical_psychology_2
from quant_fund.models.midwifery_studies import bench_midwifery_studies
from quant_fund.models.opticianry import bench_opticianry
from quant_fund.models.orthoptics import bench_orthoptics
from quant_fund.models.prosthetics_orthotics import bench_prosthetics_orthotics

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


def bench_midwifery_studies_family(seed: int = _SEED + 58100) -> dict[str, float]:
    return _finite_blob(bench_midwifery_studies(seed))


def bench_orthoptics_family(seed: int = _SEED + 58101) -> dict[str, float]:
    return _finite_blob(bench_orthoptics(seed))


def bench_audiology_studies_family(seed: int = _SEED + 58102) -> dict[str, float]:
    return _finite_blob(bench_audiology_studies(seed))


def bench_opticianry_family(seed: int = _SEED + 58103) -> dict[str, float]:
    return _finite_blob(bench_opticianry(seed))


def bench_prosthetics_orthotics_family(seed: int = _SEED + 58104) -> dict[str, float]:
    return _finite_blob(bench_prosthetics_orthotics(seed))


def bench_clinical_psychology_2_family(seed: int = _SEED + 58105) -> dict[str, float]:
    return _finite_blob(bench_clinical_psychology_2(seed))
