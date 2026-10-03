"""Wave-970 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w970 import (
    bench_fusion_algebra_family,
    bench_paragroup_family,
    bench_planar_algebra_family,
    bench_principal_graph_family,
    bench_standard_invariant_family,
    bench_subfactor_family,
)

_FAMILY_BENCHES = [
    bench_subfactor_family,
    bench_standard_invariant_family,
    bench_planar_algebra_family,
    bench_paragroup_family,
    bench_principal_graph_family,
    bench_fusion_algebra_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
