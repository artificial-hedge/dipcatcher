"""Wave-1254 model tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.allele_studies import bench_allele_studies
from quant_fund.models.cnv_studies import bench_cnv_studies
from quant_fund.models.haplotype_studies import bench_haplotype_studies
from quant_fund.models.pedigree_studies import bench_pedigree_studies
from quant_fund.models.penetrance_studies import bench_penetrance_studies
from quant_fund.models.snp_studies import bench_snp_studies


def test_allele_studies() -> None:
    assert bench_allele_studies()["synthetic_allele_studies"] == 1.0


def test_snp_studies() -> None:
    assert bench_snp_studies()["synthetic_snp_studies"] == 1.0


def test_cnv_studies() -> None:
    assert bench_cnv_studies()["synthetic_cnv_studies"] == 1.0


def test_haplotype_studies() -> None:
    assert bench_haplotype_studies()["synthetic_haplotype_studies"] == 1.0


def test_penetrance_studies() -> None:
    assert bench_penetrance_studies()["synthetic_penetrance_studies"] == 1.0


def test_pedigree_studies() -> None:
    assert bench_pedigree_studies()["synthetic_pedigree_studies"] == 1.0
