"""Wave-1238 vascular-medicine bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.aortic_medicine_studies import bench_aortic_medicine_studies
from quant_fund.models.lymphatic_medicine import bench_lymphatic_medicine
from quant_fund.models.peripheral_artery_studies import bench_peripheral_artery_studies
from quant_fund.models.phlebology_studies import bench_phlebology_studies
from quant_fund.models.vascular_lab_studies import bench_vascular_lab_studies
from quant_fund.models.vascular_medicine_studies import bench_vascular_medicine_studies

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


def bench_vascular_medicine_studies_family(seed: int = _SEED + 62600) -> dict:
    return _floats(bench_vascular_medicine_studies(seed))


def bench_phlebology_studies_family(seed: int = _SEED + 62601) -> dict:
    return _floats(bench_phlebology_studies(seed))


def bench_lymphatic_medicine_family(seed: int = _SEED + 62602) -> dict:
    return _floats(bench_lymphatic_medicine(seed))


def bench_vascular_lab_studies_family(seed: int = _SEED + 62603) -> dict:
    return _floats(bench_vascular_lab_studies(seed))


def bench_peripheral_artery_studies_family(seed: int = _SEED + 62604) -> dict:
    return _floats(bench_peripheral_artery_studies(seed))


def bench_aortic_medicine_studies_family(seed: int = _SEED + 62605) -> dict:
    return _floats(bench_aortic_medicine_studies(seed))
