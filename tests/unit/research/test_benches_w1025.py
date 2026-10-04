"""Wave-1025 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1025 import (
    bench_behavioral_econ_family,
    bench_cognitive_science_family,
    bench_game_theory2_family,
    bench_linguistics_family,
    bench_political_science_family,
    bench_sociology_net_family,
)

_FAMILY_BENCHES = [
    bench_game_theory2_family,
    bench_behavioral_econ_family,
    bench_political_science_family,
    bench_sociology_net_family,
    bench_cognitive_science_family,
    bench_linguistics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
