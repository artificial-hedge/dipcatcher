"""Wave-1250 ent-head-neck bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.cochlear_studies import bench_cochlear_studies
from quant_fund.models.head_neck_surgery_studies import bench_head_neck_surgery_studies
from quant_fund.models.laryngology_studies import bench_laryngology_studies
from quant_fund.models.otology_studies import bench_otology_studies
from quant_fund.models.rhinology_studies import bench_rhinology_studies
from quant_fund.models.sinus_studies import bench_sinus_studies

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


def bench_sinus_studies_family(seed: int = _SEED + 63800) -> dict:
    return _floats(bench_sinus_studies(seed))


def bench_laryngology_studies_family(seed: int = _SEED + 63801) -> dict:
    return _floats(bench_laryngology_studies(seed))


def bench_otology_studies_family(seed: int = _SEED + 63802) -> dict:
    return _floats(bench_otology_studies(seed))


def bench_rhinology_studies_family(seed: int = _SEED + 63803) -> dict:
    return _floats(bench_rhinology_studies(seed))


def bench_head_neck_surgery_studies_family(seed: int = _SEED + 63804) -> dict:
    return _floats(bench_head_neck_surgery_studies(seed))


def bench_cochlear_studies_family(seed: int = _SEED + 63805) -> dict:
    return _floats(bench_cochlear_studies(seed))
