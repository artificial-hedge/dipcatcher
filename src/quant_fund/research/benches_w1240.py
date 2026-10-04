"""Wave-1240 pulmonology bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.asthma_studies import bench_asthma_studies
from quant_fund.models.bronchiectasis_studies import bench_bronchiectasis_studies
from quant_fund.models.copd_studies import bench_copd_studies
from quant_fund.models.interstitial_lung_studies import bench_interstitial_lung_studies
from quant_fund.models.respiratory_studies import bench_respiratory_studies
from quant_fund.models.sleep_breathing_studies import bench_sleep_breathing_studies

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


def bench_respiratory_studies_family(seed: int = _SEED + 62800) -> dict:
    return _floats(bench_respiratory_studies(seed))


def bench_asthma_studies_family(seed: int = _SEED + 62801) -> dict:
    return _floats(bench_asthma_studies(seed))


def bench_copd_studies_family(seed: int = _SEED + 62802) -> dict:
    return _floats(bench_copd_studies(seed))


def bench_interstitial_lung_studies_family(seed: int = _SEED + 62803) -> dict:
    return _floats(bench_interstitial_lung_studies(seed))


def bench_sleep_breathing_studies_family(seed: int = _SEED + 62804) -> dict:
    return _floats(bench_sleep_breathing_studies(seed))


def bench_bronchiectasis_studies_family(seed: int = _SEED + 62805) -> dict:
    return _floats(bench_bronchiectasis_studies(seed))
