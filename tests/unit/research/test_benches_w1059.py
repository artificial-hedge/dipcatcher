"""Wave-1059 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1059 import (
    bench_ancient_history_family,
    bench_economic_history_family,
    bench_historiography_family,
    bench_intellectual_history_family,
    bench_medieval_history_family,
    bench_modern_history_family,
)

_FAMILY_BENCHES = [
    bench_historiography_family,
    bench_ancient_history_family,
    bench_medieval_history_family,
    bench_modern_history_family,
    bench_economic_history_family,
    bench_intellectual_history_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
