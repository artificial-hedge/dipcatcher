"""Wave-1245 oncology-subspecialty bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.breast_oncology_studies import bench_breast_oncology_studies
from quant_fund.models.gi_oncology_studies import bench_gi_oncology_studies
from quant_fund.models.immuno_oncology_studies import bench_immuno_oncology_studies
from quant_fund.models.medical_oncology_studies import bench_medical_oncology_studies
from quant_fund.models.targeted_therapy_studies import bench_targeted_therapy_studies
from quant_fund.models.thoracic_oncology_studies import bench_thoracic_oncology_studies

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict) -> dict:
    out = {}
    for k, v in blob.items():
        if k in _FORBIDDEN:
            raise ValueError(f"forbidden metric key {k}")
        if not k.startswith("synthetic_"):
            raise ValueError(f"non-synthetic metric key {k}")
        if not (isinstance(v, float) and 0.0 <= v <= 1.0):
            raise ValueError(f"metric {k} is not a [0,1] float")
        out[k] = v
    return out


def _floats(blob: dict) -> dict:
    return _finite_blob(blob)


def bench_medical_oncology_studies_family(seed: int = _SEED + 63300) -> dict:
    return _floats(bench_medical_oncology_studies(seed))


def bench_immuno_oncology_studies_family(seed: int = _SEED + 63301) -> dict:
    return _floats(bench_immuno_oncology_studies(seed))


def bench_targeted_therapy_studies_family(seed: int = _SEED + 63302) -> dict:
    return _floats(bench_targeted_therapy_studies(seed))


def bench_breast_oncology_studies_family(seed: int = _SEED + 63303) -> dict:
    return _floats(bench_breast_oncology_studies(seed))


def bench_thoracic_oncology_studies_family(seed: int = _SEED + 63304) -> dict:
    return _floats(bench_thoracic_oncology_studies(seed))


def bench_gi_oncology_studies_family(seed: int = _SEED + 63305) -> dict:
    return _floats(bench_gi_oncology_studies(seed))
