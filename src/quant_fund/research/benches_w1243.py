"""Wave-1243 infectious-medicine bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.healthcare_infection_studies import bench_healthcare_infection_studies
from quant_fund.models.mycosis_studies import bench_mycosis_studies
from quant_fund.models.opportunistic_studies import bench_opportunistic_studies
from quant_fund.models.sepsis_studies import bench_sepsis_studies
from quant_fund.models.sexually_transmitted_studies import bench_sexually_transmitted_studies
from quant_fund.models.tuberculosis_studies import bench_tuberculosis_studies

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


def bench_sepsis_studies_family(seed: int = _SEED + 63100) -> dict:
    return _floats(bench_sepsis_studies(seed))


def bench_tuberculosis_studies_family(seed: int = _SEED + 63101) -> dict:
    return _floats(bench_tuberculosis_studies(seed))


def bench_mycosis_studies_family(seed: int = _SEED + 63102) -> dict:
    return _floats(bench_mycosis_studies(seed))


def bench_sexually_transmitted_studies_family(seed: int = _SEED + 63103) -> dict:
    return _floats(bench_sexually_transmitted_studies(seed))


def bench_healthcare_infection_studies_family(seed: int = _SEED + 63104) -> dict:
    return _floats(bench_healthcare_infection_studies(seed))


def bench_opportunistic_studies_family(seed: int = _SEED + 63105) -> dict:
    return _floats(bench_opportunistic_studies(seed))
