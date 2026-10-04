"""Wave-1062 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1062 import (
    bench_biblical_studies_family,
    bench_buddhist_studies_family,
    bench_comparative_religion_family,
    bench_islamic_studies_family,
    bench_religious_ethics_family,
    bench_theology_family,
)

_FAMILY_BENCHES = [
    bench_theology_family,
    bench_comparative_religion_family,
    bench_biblical_studies_family,
    bench_islamic_studies_family,
    bench_buddhist_studies_family,
    bench_religious_ethics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
