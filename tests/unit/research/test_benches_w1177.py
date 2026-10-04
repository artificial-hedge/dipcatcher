"""Wave-1177 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1177 import (
    bench_axiomatic_systems_family,
    bench_formal_ontology_family,
    bench_formal_sciences_family,
    bench_mathematical_logic_family,
    bench_model_checking_2_family,
    bench_proof_calculus_family,
)

_FAMILY_BENCHES = [
    bench_formal_sciences_family,
    bench_mathematical_logic_family,
    bench_axiomatic_systems_family,
    bench_proof_calculus_family,
    bench_model_checking_2_family,
    bench_formal_ontology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
