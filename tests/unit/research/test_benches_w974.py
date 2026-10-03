"""Wave-974 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w974 import (
    bench_complex_interp_family,
    bench_lorentz_space_family,
    bench_marcinkiewicz_interp_family,
    bench_peetre_kfunctor_family,
    bench_real_interp_k_family,
    bench_reiteration_thm_family,
)

_FAMILY_BENCHES = [
    bench_real_interp_k_family,
    bench_complex_interp_family,
    bench_lorentz_space_family,
    bench_marcinkiewicz_interp_family,
    bench_peetre_kfunctor_family,
    bench_reiteration_thm_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
