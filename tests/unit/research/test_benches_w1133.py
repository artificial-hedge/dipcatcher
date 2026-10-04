"""Wave-1133 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1133 import (
    bench_conformal_field_theory_family,
    bench_holography_ads_family,
    bench_lattice_field_theory_family,
    bench_loop_quantum_gravity_family,
    bench_statistical_field_theory_family,
    bench_string_theory_math_family,
)

_FAMILY_BENCHES = [
    bench_statistical_field_theory_family,
    bench_conformal_field_theory_family,
    bench_lattice_field_theory_family,
    bench_string_theory_math_family,
    bench_loop_quantum_gravity_family,
    bench_holography_ads_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
