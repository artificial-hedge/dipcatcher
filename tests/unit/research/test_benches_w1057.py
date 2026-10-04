"""Wave-1057 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1057 import (
    bench_morphology_family,
    bench_phonetics_family,
    bench_phonology_family,
    bench_pragmatics_family,
    bench_semantics_family,
    bench_syntax_theory_family,
)

_FAMILY_BENCHES = [
    bench_phonetics_family,
    bench_phonology_family,
    bench_morphology_family,
    bench_syntax_theory_family,
    bench_semantics_family,
    bench_pragmatics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
