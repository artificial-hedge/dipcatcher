"""Wave-1256 public-health-2 bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.community_health_studies import bench_community_health_studies
from quant_fund.models.health_disparities_studies import bench_health_disparities_studies
from quant_fund.models.outbreak_studies import bench_outbreak_studies
from quant_fund.models.screening_studies import bench_screening_studies
from quant_fund.models.surveillance_studies import bench_surveillance_studies
from quant_fund.models.vaccination_studies import bench_vaccination_studies

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


def bench_screening_studies_family(seed: int = _SEED + 64400) -> dict:
    return _floats(bench_screening_studies(seed))


def bench_vaccination_studies_family(seed: int = _SEED + 64401) -> dict:
    return _floats(bench_vaccination_studies(seed))


def bench_outbreak_studies_family(seed: int = _SEED + 64402) -> dict:
    return _floats(bench_outbreak_studies(seed))


def bench_surveillance_studies_family(seed: int = _SEED + 64403) -> dict:
    return _floats(bench_surveillance_studies(seed))


def bench_health_disparities_studies_family(seed: int = _SEED + 64404) -> dict:
    return _floats(bench_health_disparities_studies(seed))


def bench_community_health_studies_family(seed: int = _SEED + 64405) -> dict:
    return _floats(bench_community_health_studies(seed))
