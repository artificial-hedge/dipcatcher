"""Wave-1127 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1127 import (
    bench_digital_history_family,
    bench_environmental_history_family,
    bench_global_history_family,
    bench_maritime_history_family,
    bench_oral_history_family,
    bench_public_history_family,
)

_FAMILY_BENCHES = [
    bench_oral_history_family,
    bench_public_history_family,
    bench_digital_history_family,
    bench_environmental_history_family,
    bench_global_history_family,
    bench_maritime_history_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
