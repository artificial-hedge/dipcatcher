"""Wave-1176 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1176 import (
    bench_biblical_studies_2_family,
    bench_buddhist_studies_2_family,
    bench_comparative_religion_2_family,
    bench_islamic_studies_2_family,
    bench_religious_studies_3_family,
    bench_theology_3_family,
)

_FAMILY_BENCHES = [
    bench_theology_3_family,
    bench_religious_studies_3_family,
    bench_comparative_religion_2_family,
    bench_biblical_studies_2_family,
    bench_islamic_studies_2_family,
    bench_buddhist_studies_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
