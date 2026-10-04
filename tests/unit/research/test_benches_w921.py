"""Wave-921 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w921 import (
    bench_atomic_bcast_family,
    bench_avalanche_consensus_family,
    bench_honey_badger_family,
    bench_isis_bcast_family,
    bench_snowball_consensus_family,
    bench_virtual_synchrony_family,
)

_FAMILY_BENCHES = [
    bench_virtual_synchrony_family,
    bench_isis_bcast_family,
    bench_atomic_bcast_family,
    bench_honey_badger_family,
    bench_avalanche_consensus_family,
    bench_snowball_consensus_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
