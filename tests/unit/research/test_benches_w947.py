"""Wave-947 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w947 import (
    bench_bezout_matrix_family,
    bench_cauchy_interlace_family,
    bench_haynsworth_inertia_family,
    bench_min_max_eig_family,
    bench_sturm_sequence_family,
    bench_sylvester_law_family,
)

_FAMILY_BENCHES = [
    bench_cauchy_interlace_family,
    bench_sylvester_law_family,
    bench_haynsworth_inertia_family,
    bench_min_max_eig_family,
    bench_sturm_sequence_family,
    bench_bezout_matrix_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
