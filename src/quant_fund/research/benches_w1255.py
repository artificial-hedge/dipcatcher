"""Wave-1255 neurodegeneration bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.als_studies import bench_als_studies
from quant_fund.models.alzheimer_studies import bench_alzheimer_studies
from quant_fund.models.dementia_studies import bench_dementia_studies
from quant_fund.models.huntington_studies import bench_huntington_studies
from quant_fund.models.ms_studies import bench_ms_studies
from quant_fund.models.parkinson_studies import bench_parkinson_studies

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


def bench_parkinson_studies_family(seed: int = _SEED + 64300) -> dict:
    return _floats(bench_parkinson_studies(seed))


def bench_alzheimer_studies_family(seed: int = _SEED + 64301) -> dict:
    return _floats(bench_alzheimer_studies(seed))


def bench_ms_studies_family(seed: int = _SEED + 64302) -> dict:
    return _floats(bench_ms_studies(seed))


def bench_als_studies_family(seed: int = _SEED + 64303) -> dict:
    return _floats(bench_als_studies(seed))


def bench_huntington_studies_family(seed: int = _SEED + 64304) -> dict:
    return _floats(bench_huntington_studies(seed))


def bench_dementia_studies_family(seed: int = _SEED + 64305) -> dict:
    return _floats(bench_dementia_studies(seed))
