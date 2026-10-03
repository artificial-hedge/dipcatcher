"""Wave-985 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w985 import (
    bench_atomic_h1_family,
    bench_bmo_space_family,
    bench_carleson_measure_family,
    bench_fefferman_stein_family,
    bench_hardy_h1_family,
    bench_john_nirenberg_family,
)

_FAMILY_BENCHES = [
    bench_hardy_h1_family,
    bench_bmo_space_family,
    bench_atomic_h1_family,
    bench_carleson_measure_family,
    bench_john_nirenberg_family,
    bench_fefferman_stein_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
