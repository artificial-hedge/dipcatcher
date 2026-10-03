"""Wave-943 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w943 import (
    bench_gershgorin_disc_family,
    bench_kadison_ineq_family,
    bench_loewner_matrix_family,
    bench_operator_convex_family,
    bench_ostrowski_bound_family,
    bench_wielandt_ineq_family,
)

_FAMILY_BENCHES = [
    bench_loewner_matrix_family,
    bench_operator_convex_family,
    bench_kadison_ineq_family,
    bench_wielandt_ineq_family,
    bench_ostrowski_bound_family,
    bench_gershgorin_disc_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
