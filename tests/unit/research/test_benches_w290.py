"""Adapter tests for wave-290 chem-informatics canon benches."""

from quant_fund.research.benches_w290 import (
    bench_mol_descriptors_family,
    bench_morgan_fp_family,
    bench_ring_detect_family,
    bench_smiles_parse_family,
    bench_substruct_family,
    bench_tanimoto_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_smiles_parse_family,
        bench_morgan_fp_family,
        bench_tanimoto_family,
        bench_mol_descriptors_family,
        bench_substruct_family,
        bench_ring_detect_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
