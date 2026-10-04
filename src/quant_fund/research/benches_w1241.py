"""Wave-1241 hematology bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.anemia_studies import bench_anemia_studies
from quant_fund.models.bleeding_disorders import bench_bleeding_disorders
from quant_fund.models.coagulation_studies import bench_coagulation_studies
from quant_fund.models.hemoglobin_studies import bench_hemoglobin_studies
from quant_fund.models.marrow_studies import bench_marrow_studies
from quant_fund.models.thrombosis_medicine import bench_thrombosis_medicine

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


def bench_anemia_studies_family(seed: int = _SEED + 62900) -> dict:
    return _floats(bench_anemia_studies(seed))


def bench_coagulation_studies_family(seed: int = _SEED + 62901) -> dict:
    return _floats(bench_coagulation_studies(seed))


def bench_hemoglobin_studies_family(seed: int = _SEED + 62902) -> dict:
    return _floats(bench_hemoglobin_studies(seed))


def bench_thrombosis_medicine_family(seed: int = _SEED + 62903) -> dict:
    return _floats(bench_thrombosis_medicine(seed))


def bench_bleeding_disorders_family(seed: int = _SEED + 62904) -> dict:
    return _floats(bench_bleeding_disorders(seed))


def bench_marrow_studies_family(seed: int = _SEED + 62905) -> dict:
    return _floats(bench_marrow_studies(seed))
