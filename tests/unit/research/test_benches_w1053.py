"""Wave-1053 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1053 import (
    bench_behavioral_neuroscience_family,
    bench_clinical_psychology_family,
    bench_cognitive_psychology_family,
    bench_developmental_psychology_family,
    bench_psychometrics_family,
    bench_social_psychology_family,
)

_FAMILY_BENCHES = [
    bench_cognitive_psychology_family,
    bench_psychometrics_family,
    bench_behavioral_neuroscience_family,
    bench_social_psychology_family,
    bench_developmental_psychology_family,
    bench_clinical_psychology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
