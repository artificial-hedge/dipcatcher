"""Wave-1249 dermatology-clinical bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.acne_studies import bench_acne_studies
from quant_fund.models.alopecia_studies import bench_alopecia_studies
from quant_fund.models.eczema_studies import bench_eczema_studies
from quant_fund.models.psoriasis_studies import bench_psoriasis_studies
from quant_fund.models.skin_cancer_studies import bench_skin_cancer_studies
from quant_fund.models.vitiligo_studies import bench_vitiligo_studies

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


def bench_skin_cancer_studies_family(seed: int = _SEED + 63700) -> dict:
    return _floats(bench_skin_cancer_studies(seed))


def bench_psoriasis_studies_family(seed: int = _SEED + 63701) -> dict:
    return _floats(bench_psoriasis_studies(seed))


def bench_eczema_studies_family(seed: int = _SEED + 63702) -> dict:
    return _floats(bench_eczema_studies(seed))


def bench_acne_studies_family(seed: int = _SEED + 63703) -> dict:
    return _floats(bench_acne_studies(seed))


def bench_vitiligo_studies_family(seed: int = _SEED + 63704) -> dict:
    return _floats(bench_vitiligo_studies(seed))


def bench_alopecia_studies_family(seed: int = _SEED + 63705) -> dict:
    return _floats(bench_alopecia_studies(seed))
