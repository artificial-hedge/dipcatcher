"""Wave-1056 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1056 import (
    bench_comparative_politics_family,
    bench_electoral_systems_family,
    bench_international_relations_family,
    bench_political_economy_family,
    bench_political_theory_family,
    bench_public_administration_family,
)

_FAMILY_BENCHES = [
    bench_comparative_politics_family,
    bench_international_relations_family,
    bench_political_theory_family,
    bench_public_administration_family,
    bench_political_economy_family,
    bench_electoral_systems_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
