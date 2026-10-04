"""Wave-1079 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1079 import (
    bench_ancient_greek_family,
    bench_classical_archaeology_family,
    bench_classical_studies_family,
    bench_latin_language_family,
    bench_papyrology_family,
    bench_philology_family,
)

_FAMILY_BENCHES = [
    bench_classical_studies_family,
    bench_latin_language_family,
    bench_ancient_greek_family,
    bench_classical_archaeology_family,
    bench_philology_family,
    bench_papyrology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
