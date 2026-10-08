"""Wave-1251 urology-andrology bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.andrology_studies import bench_andrology_studies
from quant_fund.models.bladder_studies import bench_bladder_studies
from quant_fund.models.bph_studies import bench_bph_studies
from quant_fund.models.erectile_studies import bench_erectile_studies
from quant_fund.models.incontinence_studies import bench_incontinence_studies
from quant_fund.models.prostate_studies import bench_prostate_studies

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


def bench_prostate_studies_family(seed: int = _SEED + 63900) -> dict:
    return _floats(bench_prostate_studies(seed))


def bench_bladder_studies_family(seed: int = _SEED + 63901) -> dict:
    return _floats(bench_bladder_studies(seed))


def bench_andrology_studies_family(seed: int = _SEED + 63902) -> dict:
    return _floats(bench_andrology_studies(seed))


def bench_erectile_studies_family(seed: int = _SEED + 63903) -> dict:
    return _floats(bench_erectile_studies(seed))


def bench_incontinence_studies_family(seed: int = _SEED + 63904) -> dict:
    return _floats(bench_incontinence_studies(seed))


def bench_bph_studies_family(seed: int = _SEED + 63905) -> dict:
    return _floats(bench_bph_studies(seed))
