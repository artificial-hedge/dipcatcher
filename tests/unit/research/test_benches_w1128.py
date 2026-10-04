"""Wave-1128 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1128 import (
    bench_computational_stylistics_family,
    bench_corpus_phonology_family,
    bench_language_documentation_family,
    bench_lexical_semantics_family,
    bench_stylistics_family,
    bench_translation_technology_family,
)

_FAMILY_BENCHES = [
    bench_lexical_semantics_family,
    bench_computational_stylistics_family,
    bench_stylistics_family,
    bench_corpus_phonology_family,
    bench_language_documentation_family,
    bench_translation_technology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
