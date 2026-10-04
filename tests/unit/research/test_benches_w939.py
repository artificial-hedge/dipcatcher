"""Wave-939 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w939 import (
    bench_cq_algorithm_family,
    bench_dykstra_proj_family,
    bench_halpern_iter_family,
    bench_haugazeau_proj_family,
    bench_parallel_prox_family,
    bench_split_feasibility_family,
)

_FAMILY_BENCHES = [
    bench_split_feasibility_family,
    bench_cq_algorithm_family,
    bench_dykstra_proj_family,
    bench_haugazeau_proj_family,
    bench_parallel_prox_family,
    bench_halpern_iter_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
