"""Wave-932 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w932 import (
    bench_guided_local_family,
    bench_large_neighborhood_family,
    bench_path_relinking_family,
    bench_ruin_recreate_family,
    bench_simulated_annealing_family,
    bench_vns_search_family,
)

_FAMILY_BENCHES = [
    bench_vns_search_family,
    bench_large_neighborhood_family,
    bench_ruin_recreate_family,
    bench_path_relinking_family,
    bench_guided_local_family,
    bench_simulated_annealing_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
