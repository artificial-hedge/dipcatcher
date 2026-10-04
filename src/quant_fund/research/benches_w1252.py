"""Wave-1252 metabolic-endocrine bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.hypothalamic_studies import bench_hypothalamic_studies
from quant_fund.models.lipid_studies import bench_lipid_studies
from quant_fund.models.metabolic_syndrome_studies import bench_metabolic_syndrome_studies
from quant_fund.models.obesity_studies import bench_obesity_studies
from quant_fund.models.parathyroid_studies import bench_parathyroid_studies
from quant_fund.models.pituitary_studies import bench_pituitary_studies

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


def bench_pituitary_studies_family(seed: int = _SEED + 64000) -> dict:
    return _floats(bench_pituitary_studies(seed))


def bench_parathyroid_studies_family(seed: int = _SEED + 64001) -> dict:
    return _floats(bench_parathyroid_studies(seed))


def bench_lipid_studies_family(seed: int = _SEED + 64002) -> dict:
    return _floats(bench_lipid_studies(seed))


def bench_obesity_studies_family(seed: int = _SEED + 64003) -> dict:
    return _floats(bench_obesity_studies(seed))


def bench_metabolic_syndrome_studies_family(seed: int = _SEED + 64004) -> dict:
    return _floats(bench_metabolic_syndrome_studies(seed))


def bench_hypothalamic_studies_family(seed: int = _SEED + 64005) -> dict:
    return _floats(bench_hypothalamic_studies(seed))
