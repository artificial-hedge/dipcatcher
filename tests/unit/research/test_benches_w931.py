"""Wave-931 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w931 import (
    bench_ant_colony_family,
    bench_diff_evolution_family,
    bench_firefly_algo_family,
    bench_genetic_tsp_family,
    bench_harmony_search_family,
    bench_pso_swarm_family,
)

_FAMILY_BENCHES = [
    bench_ant_colony_family,
    bench_pso_swarm_family,
    bench_diff_evolution_family,
    bench_genetic_tsp_family,
    bench_firefly_algo_family,
    bench_harmony_search_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
