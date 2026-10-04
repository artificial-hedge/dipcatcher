"""Wave-1242 rehab-medicine bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.fracture_studies import bench_fracture_studies
from quant_fund.models.osteoporosis_studies import bench_osteoporosis_studies
from quant_fund.models.physiatry_studies import bench_physiatry_studies
from quant_fund.models.physical_therapy_studies import bench_physical_therapy_studies
from quant_fund.models.rehabilitation_studies import bench_rehabilitation_studies
from quant_fund.models.sports_injury_studies import bench_sports_injury_studies

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


def bench_rehabilitation_studies_family(seed: int = _SEED + 63000) -> dict:
    return _floats(bench_rehabilitation_studies(seed))


def bench_physical_therapy_studies_family(seed: int = _SEED + 63001) -> dict:
    return _floats(bench_physical_therapy_studies(seed))


def bench_sports_injury_studies_family(seed: int = _SEED + 63002) -> dict:
    return _floats(bench_sports_injury_studies(seed))


def bench_fracture_studies_family(seed: int = _SEED + 63003) -> dict:
    return _floats(bench_fracture_studies(seed))


def bench_osteoporosis_studies_family(seed: int = _SEED + 63004) -> dict:
    return _floats(bench_osteoporosis_studies(seed))


def bench_physiatry_studies_family(seed: int = _SEED + 63005) -> dict:
    return _floats(bench_physiatry_studies(seed))
