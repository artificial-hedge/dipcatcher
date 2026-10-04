"""Wave-1099 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1099 import (
    bench_abnormal_psychology_family,
    bench_forensic_psychology_family,
    bench_health_psychology_family,
    bench_neuropsychology_family,
    bench_organizational_psychology_family,
    bench_personality_psychology_family,
)

_FAMILY_BENCHES = [
    bench_personality_psychology_family,
    bench_abnormal_psychology_family,
    bench_health_psychology_family,
    bench_neuropsychology_family,
    bench_forensic_psychology_family,
    bench_organizational_psychology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
