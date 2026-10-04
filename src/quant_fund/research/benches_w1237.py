"""Wave-1237 clinical-pharmacy bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.clinical_pharmacy_studies import bench_clinical_pharmacy_studies
from quant_fund.models.compounding_pharmacy import bench_compounding_pharmacy
from quant_fund.models.hospital_pharmacy_studies import bench_hospital_pharmacy_studies
from quant_fund.models.medication_therapy_mgmt import bench_medication_therapy_mgmt
from quant_fund.models.pharmacovigilance_studies import bench_pharmacovigilance_studies
from quant_fund.models.pharmacy_practice_studies import bench_pharmacy_practice_studies

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231


def _finite_blob(blob: dict) -> dict:
    out = {}
    for k, v in blob.items():
        assert k not in _FORBIDDEN
        assert k.startswith("synthetic_")
        assert isinstance(v, float) and 0.0 <= v <= 1.0
        out[k] = v
    return out


def _floats(blob: dict) -> dict:
    return _finite_blob(blob)


def bench_clinical_pharmacy_studies_family(seed: int = _SEED + 62500) -> dict:
    return _floats(bench_clinical_pharmacy_studies(seed))


def bench_pharmacy_practice_studies_family(seed: int = _SEED + 62501) -> dict:
    return _floats(bench_pharmacy_practice_studies(seed))


def bench_medication_therapy_mgmt_family(seed: int = _SEED + 62502) -> dict:
    return _floats(bench_medication_therapy_mgmt(seed))


def bench_compounding_pharmacy_family(seed: int = _SEED + 62503) -> dict:
    return _floats(bench_compounding_pharmacy(seed))


def bench_pharmacovigilance_studies_family(seed: int = _SEED + 62504) -> dict:
    return _floats(bench_pharmacovigilance_studies(seed))


def bench_hospital_pharmacy_studies_family(seed: int = _SEED + 62505) -> dict:
    return _floats(bench_hospital_pharmacy_studies(seed))
