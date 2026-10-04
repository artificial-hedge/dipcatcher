"""Wave-1209 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1209 import (
    bench_addiction_medicine_family,
    bench_community_psychiatry_family,
    bench_consultation_liaison_family,
    bench_eating_disorders_family,
    bench_psychosomatic_medicine_family,
    bench_sleep_disorders_family,
)

_FAMILY_BENCHES = [
    bench_addiction_medicine_family,
    bench_eating_disorders_family,
    bench_sleep_disorders_family,
    bench_psychosomatic_medicine_family,
    bench_consultation_liaison_family,
    bench_community_psychiatry_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
