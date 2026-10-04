"""Wave-1038 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1038 import (
    bench_cardiology_family,
    bench_human_physiology_family,
    bench_immunology_family,
    bench_neuroscience_med_family,
    bench_pathology_family,
    bench_pharmacokinetics_family,
)

_FAMILY_BENCHES = [
    bench_human_physiology_family,
    bench_pharmacokinetics_family,
    bench_immunology_family,
    bench_pathology_family,
    bench_neuroscience_med_family,
    bench_cardiology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
