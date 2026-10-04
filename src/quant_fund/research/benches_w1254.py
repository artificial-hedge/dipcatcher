"""Wave-1254 molecular-genetics-2 bench adapters (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.allele_studies import bench_allele_studies
from quant_fund.models.cnv_studies import bench_cnv_studies
from quant_fund.models.haplotype_studies import bench_haplotype_studies
from quant_fund.models.pedigree_studies import bench_pedigree_studies
from quant_fund.models.penetrance_studies import bench_penetrance_studies
from quant_fund.models.snp_studies import bench_snp_studies

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


def bench_allele_studies_family(seed: int = _SEED + 64200) -> dict:
    return _floats(bench_allele_studies(seed))


def bench_snp_studies_family(seed: int = _SEED + 64201) -> dict:
    return _floats(bench_snp_studies(seed))


def bench_cnv_studies_family(seed: int = _SEED + 64202) -> dict:
    return _floats(bench_cnv_studies(seed))


def bench_haplotype_studies_family(seed: int = _SEED + 64203) -> dict:
    return _floats(bench_haplotype_studies(seed))


def bench_penetrance_studies_family(seed: int = _SEED + 64204) -> dict:
    return _floats(bench_penetrance_studies(seed))


def bench_pedigree_studies_family(seed: int = _SEED + 64205) -> dict:
    return _floats(bench_pedigree_studies(seed))
