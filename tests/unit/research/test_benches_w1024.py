"""Wave-1024 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1024 import (
    bench_dna_sequencing_family,
    bench_gene_expression_family,
    bench_metabolomics_family,
    bench_phylogenetics_family,
    bench_protein_folding_family,
    bench_systems_biology_family,
)

_FAMILY_BENCHES = [
    bench_protein_folding_family,
    bench_dna_sequencing_family,
    bench_phylogenetics_family,
    bench_gene_expression_family,
    bench_metabolomics_family,
    bench_systems_biology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
