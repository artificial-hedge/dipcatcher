"""Wave-1246 surgical-subspecialty bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.bariatric_surgery_studies import bench_bariatric_surgery_studies
from quant_fund.models.burn_surgery_studies import bench_burn_surgery_studies
from quant_fund.models.endocrine_surgery_studies import bench_endocrine_surgery_studies
from quant_fund.models.pediatric_surgery_studies import bench_pediatric_surgery_studies
from quant_fund.models.plastic_surgery_studies import bench_plastic_surgery_studies
from quant_fund.models.transplant_surgery_studies import bench_transplant_surgery_studies

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


def bench_bariatric_surgery_studies_family(seed: int = _SEED + 63400) -> dict:
    return _floats(bench_bariatric_surgery_studies(seed))


def bench_pediatric_surgery_studies_family(seed: int = _SEED + 63401) -> dict:
    return _floats(bench_pediatric_surgery_studies(seed))


def bench_plastic_surgery_studies_family(seed: int = _SEED + 63402) -> dict:
    return _floats(bench_plastic_surgery_studies(seed))


def bench_burn_surgery_studies_family(seed: int = _SEED + 63403) -> dict:
    return _floats(bench_burn_surgery_studies(seed))


def bench_endocrine_surgery_studies_family(seed: int = _SEED + 63404) -> dict:
    return _floats(bench_endocrine_surgery_studies(seed))


def bench_transplant_surgery_studies_family(seed: int = _SEED + 63405) -> dict:
    return _floats(bench_transplant_surgery_studies(seed))
