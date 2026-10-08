"""Wave-1247 nephro-renal bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.aki_studies import bench_aki_studies
from quant_fund.models.ckd_studies import bench_ckd_studies
from quant_fund.models.electrolyte_studies import bench_electrolyte_studies
from quant_fund.models.glomerular_studies import bench_glomerular_studies
from quant_fund.models.stones_studies import bench_stones_studies
from quant_fund.models.tubulointerstitial_studies import bench_tubulointerstitial_studies

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


def bench_glomerular_studies_family(seed: int = _SEED + 63500) -> dict:
    return _floats(bench_glomerular_studies(seed))


def bench_tubulointerstitial_studies_family(seed: int = _SEED + 63501) -> dict:
    return _floats(bench_tubulointerstitial_studies(seed))


def bench_ckd_studies_family(seed: int = _SEED + 63502) -> dict:
    return _floats(bench_ckd_studies(seed))


def bench_aki_studies_family(seed: int = _SEED + 63503) -> dict:
    return _floats(bench_aki_studies(seed))


def bench_electrolyte_studies_family(seed: int = _SEED + 63504) -> dict:
    return _floats(bench_electrolyte_studies(seed))


def bench_stones_studies_family(seed: int = _SEED + 63505) -> dict:
    return _floats(bench_stones_studies(seed))
