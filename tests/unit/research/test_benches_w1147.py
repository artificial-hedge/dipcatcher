"""Wave-1147 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1147 import (
    bench_anatomy_family,
    bench_cardiology_2_family,
    bench_endocrinology_2_family,
    bench_immunology_2_family,
    bench_neuroscience_2_family,
    bench_physiology_2_family,
)

_FAMILY_BENCHES = [
    bench_anatomy_family,
    bench_physiology_2_family,
    bench_endocrinology_2_family,
    bench_neuroscience_2_family,
    bench_cardiology_2_family,
    bench_immunology_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
