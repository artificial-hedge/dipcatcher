"""Wave-1244 imaging-modality bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.ct_imaging_studies import bench_ct_imaging_studies
from quant_fund.models.mammography_studies import bench_mammography_studies
from quant_fund.models.mri_studies import bench_mri_studies
from quant_fund.models.neuroradiology_studies import bench_neuroradiology_studies
from quant_fund.models.pet_imaging_studies import bench_pet_imaging_studies
from quant_fund.models.ultrasound_studies import bench_ultrasound_studies

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


def bench_neuroradiology_studies_family(seed: int = _SEED + 63200) -> dict:
    return _floats(bench_neuroradiology_studies(seed))


def bench_mammography_studies_family(seed: int = _SEED + 63201) -> dict:
    return _floats(bench_mammography_studies(seed))


def bench_ultrasound_studies_family(seed: int = _SEED + 63202) -> dict:
    return _floats(bench_ultrasound_studies(seed))


def bench_ct_imaging_studies_family(seed: int = _SEED + 63203) -> dict:
    return _floats(bench_ct_imaging_studies(seed))


def bench_mri_studies_family(seed: int = _SEED + 63204) -> dict:
    return _floats(bench_mri_studies(seed))


def bench_pet_imaging_studies_family(seed: int = _SEED + 63205) -> dict:
    return _floats(bench_pet_imaging_studies(seed))
