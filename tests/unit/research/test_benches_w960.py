"""Wave-960 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w960 import (
    bench_banach_algebra_family,
    bench_c_star_algebra_family,
    bench_gelfand_transform_family,
    bench_holomorphic_calculus_family,
    bench_positive_functional_family,
    bench_spectrum_algebra_family,
)

_FAMILY_BENCHES = [
    bench_banach_algebra_family,
    bench_gelfand_transform_family,
    bench_c_star_algebra_family,
    bench_spectrum_algebra_family,
    bench_holomorphic_calculus_family,
    bench_positive_functional_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
