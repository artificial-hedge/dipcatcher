"""Wave-1248 ophthalmology-vision bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.cataract_studies import bench_cataract_studies
from quant_fund.models.corneal_studies import bench_corneal_studies
from quant_fund.models.glaucoma_studies import bench_glaucoma_studies
from quant_fund.models.macular_studies import bench_macular_studies
from quant_fund.models.refractive_studies import bench_refractive_studies
from quant_fund.models.retinal_studies import bench_retinal_studies

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


def bench_retinal_studies_family(seed: int = _SEED + 63600) -> dict:
    return _floats(bench_retinal_studies(seed))


def bench_corneal_studies_family(seed: int = _SEED + 63601) -> dict:
    return _floats(bench_corneal_studies(seed))


def bench_glaucoma_studies_family(seed: int = _SEED + 63602) -> dict:
    return _floats(bench_glaucoma_studies(seed))


def bench_cataract_studies_family(seed: int = _SEED + 63603) -> dict:
    return _floats(bench_cataract_studies(seed))


def bench_macular_studies_family(seed: int = _SEED + 63604) -> dict:
    return _floats(bench_macular_studies(seed))


def bench_refractive_studies_family(seed: int = _SEED + 63605) -> dict:
    return _floats(bench_refractive_studies(seed))
