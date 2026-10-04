"""Wave-941 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w941 import (
    bench_augmented_lagr_family,
    bench_ekeland_var_family,
    bench_limiting_subdiff_family,
    bench_monteiro_semismooth_family,
    bench_proximal_subdiff_family,
    bench_semismooth_newton_family,
)

_FAMILY_BENCHES = [
    bench_limiting_subdiff_family,
    bench_proximal_subdiff_family,
    bench_ekeland_var_family,
    bench_monteiro_semismooth_family,
    bench_semismooth_newton_family,
    bench_augmented_lagr_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
