"""Wave-940 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w940 import (
    bench_forward_reflected_family,
    bench_korpelevich_eg_family,
    bench_popov_alg_family,
    bench_reflected_golden_family,
    bench_subgradient_extragradient_family,
    bench_tseng_fb_family,
)

_FAMILY_BENCHES = [
    bench_subgradient_extragradient_family,
    bench_korpelevich_eg_family,
    bench_popov_alg_family,
    bench_tseng_fb_family,
    bench_forward_reflected_family,
    bench_reflected_golden_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
