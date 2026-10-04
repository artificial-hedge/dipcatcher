"""Wave-1050 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1050 import (
    bench_clinical_pharmacology_family,
    bench_drug_metabolism_family,
    bench_neuropharmacology_family,
    bench_pharmacodynamics_family,
    bench_pharmacokinetics_2_family,
    bench_toxicology_family,
)

_FAMILY_BENCHES = [
    bench_pharmacodynamics_family,
    bench_pharmacokinetics_2_family,
    bench_toxicology_family,
    bench_clinical_pharmacology_family,
    bench_neuropharmacology_family,
    bench_drug_metabolism_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
