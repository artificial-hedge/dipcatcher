"""Wave-1254 bench adapter tests (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.research.benches_w1254 import (
    bench_allele_studies_family,
    bench_cnv_studies_family,
    bench_haplotype_studies_family,
    bench_pedigree_studies_family,
    bench_penetrance_studies_family,
    bench_snp_studies_family,
)

_FAMILY_BENCHES = [
    bench_allele_studies_family,
    bench_snp_studies_family,
    bench_cnv_studies_family,
    bench_haplotype_studies_family,
    bench_penetrance_studies_family,
    bench_pedigree_studies_family,
]


def test_families_emit_synthetic_scores() -> None:
    for bench in _FAMILY_BENCHES:
        blob = bench()
        assert len(blob) == 1
        k, v = next(iter(blob.items()))
        assert k.startswith("synthetic_")
        assert 0.0 <= v <= 1.0
