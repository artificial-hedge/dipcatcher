"""Wave-1239 gi-medicine bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.celiac_studies import bench_celiac_studies
from quant_fund.models.gi_endoscopy_studies import bench_gi_endoscopy_studies
from quant_fund.models.hepatology_medicine import bench_hepatology_medicine
from quant_fund.models.ibd_studies import bench_ibd_studies
from quant_fund.models.motility_studies import bench_motility_studies
from quant_fund.models.pancreatic_medicine import bench_pancreatic_medicine

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


def bench_gi_endoscopy_studies_family(seed: int = _SEED + 62700) -> dict:
    return _floats(bench_gi_endoscopy_studies(seed))


def bench_hepatology_medicine_family(seed: int = _SEED + 62701) -> dict:
    return _floats(bench_hepatology_medicine(seed))


def bench_pancreatic_medicine_family(seed: int = _SEED + 62702) -> dict:
    return _floats(bench_pancreatic_medicine(seed))


def bench_ibd_studies_family(seed: int = _SEED + 62703) -> dict:
    return _floats(bench_ibd_studies(seed))


def bench_celiac_studies_family(seed: int = _SEED + 62704) -> dict:
    return _floats(bench_celiac_studies(seed))


def bench_motility_studies_family(seed: int = _SEED + 62705) -> dict:
    return _floats(bench_motility_studies(seed))
