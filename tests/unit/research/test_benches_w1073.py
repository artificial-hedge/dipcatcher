"""Wave-1073 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1073 import (
    bench_biblical_exegesis_family,
    bench_church_history_family,
    bench_liturgical_studies_family,
    bench_missiology_family,
    bench_pastoral_theology_family,
    bench_systematic_theology_family,
)

_FAMILY_BENCHES = [
    bench_systematic_theology_family,
    bench_biblical_exegesis_family,
    bench_church_history_family,
    bench_pastoral_theology_family,
    bench_liturgical_studies_family,
    bench_missiology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
