"""Wave-922 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w922 import (
    bench_abcast_lite_family,
    bench_cap_theorem_family,
    bench_cbc_bcast_family,
    bench_lake_wisc_family,
    bench_slush_consensus_family,
    bench_snowflake_consensus_family,
)

_FAMILY_BENCHES = [
    bench_abcast_lite_family,
    bench_cbc_bcast_family,
    bench_slush_consensus_family,
    bench_snowflake_consensus_family,
    bench_cap_theorem_family,
    bench_lake_wisc_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
