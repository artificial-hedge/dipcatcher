"""Wave-1236 pain bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.chronic_pain_studies import bench_chronic_pain_studies
from quant_fund.models.fibromyalgia_studies import bench_fibromyalgia_studies
from quant_fund.models.headache_studies import bench_headache_studies
from quant_fund.models.interventional_pain_studies import bench_interventional_pain_studies
from quant_fund.models.neuropathic_pain_studies import bench_neuropathic_pain_studies
from quant_fund.models.opioid_stewardship_studies import bench_opioid_stewardship_studies

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


def bench_chronic_pain_studies_family(seed: int = _SEED + 62400) -> dict:
    return _floats(bench_chronic_pain_studies(seed))


def bench_fibromyalgia_studies_family(seed: int = _SEED + 62401) -> dict:
    return _floats(bench_fibromyalgia_studies(seed))


def bench_headache_studies_family(seed: int = _SEED + 62402) -> dict:
    return _floats(bench_headache_studies(seed))


def bench_neuropathic_pain_studies_family(seed: int = _SEED + 62403) -> dict:
    return _floats(bench_neuropathic_pain_studies(seed))


def bench_opioid_stewardship_studies_family(seed: int = _SEED + 62404) -> dict:
    return _floats(bench_opioid_stewardship_studies(seed))


def bench_interventional_pain_studies_family(seed: int = _SEED + 62405) -> dict:
    return _floats(bench_interventional_pain_studies(seed))
