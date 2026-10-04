"""Wave-1104 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1104 import (
    bench_american_politics_family,
    bench_policy_analysis_family,
    bench_political_behavior_family,
    bench_political_methodology_family,
    bench_public_law_family,
    bench_security_studies_family,
)

_FAMILY_BENCHES = [
    bench_american_politics_family,
    bench_political_behavior_family,
    bench_public_law_family,
    bench_political_methodology_family,
    bench_security_studies_family,
    bench_policy_analysis_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
